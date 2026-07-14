import os
import sys

# Safe print to prevent OSError [Errno 22] when running in headless/background redirect mode on Windows
def safe_print(*args, **kwargs):
    try:
        sys.stdout.write(" ".join(map(str, args)) + "\n")
        sys.stdout.flush()
    except OSError:
        pass

import builtins
builtins.print = safe_print

import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.config import BEST_MODEL_PATH, PREPROCESSOR_PATH, MODEL_DIR, DATA_DIR
from src.explainability import get_natural_language_explanation, explain_single_order
from src.prescriptive import (
    compute_risk_priority_index, compute_revenue_at_risk, 
    compute_sla_risk_index, compute_intervention_score, 
    get_prescriptive_recommendations
)

@st.cache_resource
def load_ml_pipeline():
    """
    Loads and caches the trained machine learning model, SHAP explainer,
    and preprocessing components.
    """
    try:
        # Load best model and preprocessor using open file descriptors for cross-platform reliability
        with open(BEST_MODEL_PATH, "rb") as f:
            model = joblib.load(f)
            
        with open(PREPROCESSOR_PATH, "rb") as f:
            pipeline = joblib.load(f)
        
        # Load SHAP explainer
        explainer_path = os.path.join(MODEL_DIR, "shap_explainer.joblib")
        if os.path.exists(explainer_path):
            with open(explainer_path, "rb") as f:
                explainer = joblib.load(f)
        else:
            explainer = None
            
        # Load SHAP values sample
        shap_values_path = os.path.join(MODEL_DIR, "shap_values_sample.joblib")
        if os.path.exists(shap_values_path):
            with open(shap_values_path, "rb") as f:
                shap_sample = joblib.load(f)
        else:
            shap_sample = None
            
        return model, pipeline, explainer, shap_sample
    except Exception as e:
        st.error(f"❌ Error loading ML pipeline: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None, None

def get_dashboard_data_sample():
    """
    Loads a sample of 2,000 test set rows to run the dashboard efficiently.
    Ensures dates are present and features are already engineered for rapid chart rendering.
    """
    sample_path = os.path.join(DATA_DIR, "dashboard_sample.csv")
    if os.path.exists(sample_path):
        print("Loading pre-processed dashboard sample...")
        sample_df = pd.read_csv(sample_path)
        sample_df["Order Date"] = pd.to_datetime(sample_df["Order Date"])
        return sample_df

    from src.data_loader import load_raw_data, preprocess_raw_data, get_train_test_split
    from src.features import LogisticsFeatureEngineer
    
    raw_df = load_raw_data()
    clean_df = preprocess_raw_data(raw_df)
    _, test_df = get_train_test_split(clean_df, split_method="temporal")
    
    # We sample 2,000 records from the test set (Nov-Dec 2025)
    sample_df = test_df.sample(n=min(2000, len(test_df)), random_state=42).copy()
    
    # Run feature engineering to populate congestion and reliability indexes
    _, pipeline, _, _ = load_ml_pipeline()
    if pipeline is not None:
        fe = pipeline["feature_engineer"]
        sample_fe = fe.transform(sample_df)
    else:
        fe = LogisticsFeatureEngineer()
        fe.fit(test_df) # fallback
        sample_fe = fe.transform(sample_df)
        
    return sample_fe

def predict_df_risks(df, model, pipeline_components):
    """
    Batch predicts delay probabilities and appends risk indexes for a dataframe.
    """
    preprocessor = pipeline_components["preprocessor"]
    
    # Prepare features for model
    from src.config import COLS_TO_DROP
    cols_to_drop = [c for c in COLS_TO_DROP if c in df.columns]
    X_raw = df.drop(columns=cols_to_drop + ["Late_delivery_risk", "Order Date"], errors="ignore")
    
    X_processed = preprocessor.transform(X_raw)
    probs = model.predict_proba(X_processed)[:, 1]
    
    df_out = df.copy()
    df_out["Risk_Probability"] = probs
    df_out["Risk_Level"] = pd.cut(
        probs, 
        bins=[-0.1, 0.3, 0.7, 1.1], 
        labels=["Low", "Medium", "High"]
    )
    
    # Calculate operational metrics
    # Complexity is already in the dataframe as 'Order Complexity Score'
    df_out["Risk_Priority_Index"] = df_out.apply(
        lambda row: compute_risk_priority_index(row["Risk_Probability"], row["Order Complexity Score"]), axis=1
    )
    df_out["Revenue_at_Risk"] = df_out.apply(
        lambda row: compute_revenue_at_risk(row["Risk_Probability"], row["Sales"]), axis=1
    )
    df_out["SLA_Risk_Index"] = df_out.apply(
        lambda row: compute_sla_risk_index(row["Risk_Probability"], row["Days for shipment (scheduled)"]), axis=1
    )
    df_out["Intervention_Score"] = df_out.apply(
        lambda row: compute_intervention_score(
            row["Risk_Probability"], row["Order Complexity Score"], row["Sales"], row["Days for shipment (scheduled)"]
        ), axis=1
    )
    
    return df_out

def handle_copilot_query(query, sample_df, model, pipeline, api_key=None):
    """
    Logistics GenAI Copilot query engine. Connects to Gemini LLM if api_key is provided,
    sending detailed logistics database context and user query. Falls back to keyword
    matching if api_key is missing.
    """
    # 1. Check if we have a Gemini API key and try calling the LLM
    if api_key:
        try:
            import google.generativeai as genai
            clean_api_key = api_key.strip().strip("'").strip('"')
            genai.configure(api_key=clean_api_key)
            
            total_orders = len(sample_df)
            
            # Risk breakdown
            risk_counts = sample_df["Risk_Level"].value_counts()
            high_risk = risk_counts.get("High", 0)
            med_risk = risk_counts.get("Medium", 0)
            low_risk = risk_counts.get("Low", 0)
            
            # Revenue at risk
            total_rar = sample_df["Revenue_at_Risk"].sum()
            high_risk_rar = sample_df[sample_df["Risk_Level"] == "High"]["Revenue_at_Risk"].sum()
            
            # Congested regions
            fe = pipeline["feature_engineer"]
            congestion_sorted = sorted(fe.regional_congestion.items(), key=lambda x: x[1], reverse=True)
            top_congested = congestion_sorted[:5]
            congested_str = "\n".join([f"- {reg}: {rate*100:.1f}% historical delay rate" for reg, rate in top_congested])
            
            # Average risk by Shipping Mode
            mode_delays = sample_df.groupby("Shipping Mode")["Risk_Probability"].mean().reset_index()
            mode_str = "\n".join([f"- {row['Shipping Mode']}: {row['Risk_Probability']*100:.1f}% average risk" for _, row in mode_delays.iterrows()])
            
            # Get a sample of the top 5 highest risk orders
            high_risk_orders = sample_df[sample_df["Risk_Level"] == "High"].sort_values(by="Risk_Probability", ascending=False).head(5)
            orders_list = []
            for _, row in high_risk_orders.iterrows():
                orders_list.append(
                    f"- Customer ID {row['Customer Id']} | Product: {row['Product Name']} | "
                    f"Destination: {row['Order City']}, {row['Order Country']} | "
                    f"Sales: ${row['Sales']:.2f} | Delay Risk: {row['Risk_Probability']*100:.1f}% | "
                    f"Intervention Score: {row.get('Intervention_Score', 0.0):.1f}"
                )
            orders_str = "\n".join(orders_list)
            
            # Build Context Prompt
            context = f"""
LOGISTICS DATABASE CONTEXT:
- Total Orders in Pending Queue: {total_orders:,}
- Risk Level Breakdown:
  * High Risk: {high_risk} ({high_risk/total_orders*100:.1f}%)
  * Medium Risk: {med_risk} ({med_risk/total_orders*100:.1f}%)
  * Low Risk: {low_risk} ({low_risk/total_orders*100:.1f}%)
- Financial Exposure (Revenue at Risk):
  * Total Revenue at Risk: ${total_rar:,.2f}
  * Revenue at Risk in High-Risk Queue: ${high_risk_rar:,.2f}
- Top Congested Shipping Regions (Historical Delay Rate):
{congested_str}
- Average predicted delay risk by Shipping Mode:
{mode_str}
- Sample of Top 5 Highest Risk Orders currently in the Queue:
{orders_str}
"""
            
            system_instruction = """
You are the "APL Logistics Supply Chain Copilot", an expert AI assistant connected to the active logistics database and predictive ML models.
Your role is to help operations managers analyze shipping delays, evaluate financial exposures, and propose actionable operational triage/interventions (e.g. shipping upgrades, prioritization flags, carrier rerouting).
Use the provided LOGISTICS DATABASE CONTEXT to answer the user's question. If the user's question asks for specific details not in the context, use your supply chain expertise to give a reasonable, helpful answer while noting any data constraints.
Always be professional, operational-focused, and direct. Format your response clearly with headers and bullet points.
"""
            
            prompt_text = f"{system_instruction}\n\n{context}\n\nUSER QUESTION: {query}\n\nRESPONSE:"
            try:
                # Try gemini-2.5-flash
                llm = genai.GenerativeModel("gemini-2.5-flash")
                response = llm.generate_content(prompt_text)
            except Exception as e1:
                try:
                    # Fallback to gemini-3.5-flash
                    llm = genai.GenerativeModel("gemini-3.5-flash")
                    response = llm.generate_content(prompt_text)
                except Exception as e2:
                    try:
                        # Fallback to gemini-flash-latest
                        llm = genai.GenerativeModel("gemini-flash-latest")
                        response = llm.generate_content(prompt_text)
                    except Exception as e3:
                        raise Exception(
                            f"Failed with gemini-2.5-flash ({str(e1)}) | "
                            f"gemini-3.5-flash ({str(e2)}) | "
                            f"gemini-flash-latest ({str(e3)})"
                        )
            return response.text
        except Exception as e:
            # Save error message to append to response
            error_msg = f"\n\n*(Note: An error occurred while communicating with Gemini API: {str(e)}. Falling back to rule-based parser)*"
            
    # 2. Fallback Keyword Matcher (Rule-Based Parser)
    query_clean = query.lower().strip()
    
    # Check if we should append the warning banner about missing/failing API key
    prefix = ""
    if not api_key:
        prefix = "⚠️ **Note**: Gemini API Key is missing. The Copilot is currently running in fallback rule-based mode. To enable the fully dynamic Generative AI Copilot, please enter your Gemini API key in the sidebar configuration.\n\n"
    elif 'error_msg' in locals():
        prefix = f"⚠️ **Error running Gemini model**: {error_msg}\n\n"
        
    # 1. Congested regions query
    if "congest" in query_clean or "slowest" in query_clean or "worst region" in query_clean or "highest risk region" in query_clean:
        fe = pipeline["feature_engineer"]
        congestion_sorted = sorted(fe.regional_congestion.items(), key=lambda x: x[1], reverse=True)
        top_congested = congestion_sorted[:3]
        
        response = prefix + "Based on our model's historical training records, the top 3 most congested order regions with the highest delay rates are:\n\n"
        for i, (reg, rate) in enumerate(top_congested):
            response += f"{i+1}. **{reg}** — Historical delay rate of **{rate*100:.1f}%**\n"
        response += "\n*Operational recommendation: For high-priority orders routing through these regions, we advise applying a warehousing priority pick flag or upgrading shipments to First Class.*"
        return response
        
    # 2. Shipping mode analysis query
    elif "shipping mode" in query_clean or "delivery mode" in query_clean or "carrier speed" in query_clean:
        mode_delays = sample_df.groupby("Shipping Mode")["Risk_Probability"].mean().reset_index()
        mode_delays = mode_delays.sort_values(by="Risk_Probability", ascending=False)
        
        response = prefix + "Here is the average predicted delay risk by Shipping Mode for the current pending queue:\n\n"
        for _, row in mode_delays.iterrows():
            response += f"• **{row['Shipping Mode']}**: **{row['Risk_Probability']*100:.1f}%** average delay risk\n"
        response += "\n*Key Insight: First Class has a extremely high delay rate (~95%) because scheduled days are capped at 1, leaving zero buffer for carrier transit variances. Upgrading to Same Day or routing through local warehouses is recommended.*"
        return response
        
    # 3. Revenue exposure query
    elif "revenue at risk" in query_clean or "financial exposure" in query_clean or "value at risk" in query_clean:
        total_rar = sample_df["Revenue_at_Risk"].sum()
        high_risk_rar = sample_df[sample_df["Risk_Level"] == "High"]["Revenue_at_Risk"].sum()
        
        response = prefix + f"### Financial Risk Exposure Summary:\n\n"
        response += f"• **Total Revenue at Risk**: **${total_rar:,.2f}**\n"
        response += f"• **Revenue at Risk in High-Risk Queue**: **${high_risk_rar:,.2f}**\n\n"
        response += "The operations team should focus triage efforts on the High-Risk queue, sorting by *Intervention Score* to mitigate the maximum amount of revenue with the lowest intervention cost."
        return response
        
    # 5. Explaining a specific order (mocked or random order id query)
    elif "order" in query_clean:
        import re
        numbers = re.findall(r'\d+', query_clean)
        if numbers:
            cust_id = int(numbers[0])
            cust_orders = sample_df[sample_df["Customer Id"] == cust_id]
            if not cust_orders.empty:
                order_row = cust_orders.iloc[0].to_dict()
                prob = order_row["Risk_Probability"]
                recs = get_prescriptive_recommendations(order_row, prob)
                
                response = prefix + f"### Order Analysis for Customer ID {cust_id}:\n\n"
                response += f"• **Predicted Delay Risk**: **{prob*100:.1f}%** ({order_row['Risk_Level']} Risk)\n"
                response += f"• **Product**: {order_row['Product Name']}\n"
                response += f"• **Destination**: {order_row['Order City']}, {order_row['Order Country']}\n"
                response += f"• **Revenue Exposure**: ${order_row['Revenue_at_Risk']:.2f}\n\n"
                response += "**Suggested Action Plan:**\n"
                for rec in recs:
                    response += f"- **{rec['Action']}** (Dept: {rec['Department']}, Cost: ${rec['Cost ($)']}, Delay Risk Reduction: -{rec['Risk Reduction (%)']}%)\n"
                return response
            
        high_risk_order = sample_df[sample_df["Risk_Level"] == "High"].iloc[0].to_dict()
        prob = high_risk_order["Risk_Probability"]
        recs = get_prescriptive_recommendations(high_risk_order, prob)
        
        response = prefix + f"I couldn't locate that specific order ID in the active batch. Here is an analysis of a typical high-risk order (Customer ID {high_risk_order['Customer Id']}) currently in the queue:\n\n"
        response += f"• **Product**: {high_risk_order['Product Name']}\n"
        response += f"• **Destination**: {high_risk_order['Order City']}, {high_risk_order['Order Country']}\n"
        response += f"• **Risk**: **{prob*100:.1f}%**\n\n"
        response += "**Recommended Interventions:**\n"
        for rec in recs:
            response += f"- **{rec['Action']}** (Est. Cost: ${rec['Cost ($)']}, Risk Reduction: -{rec['Risk Reduction (%)']}%)\n"
        return response
        
    # 6. Default Copilot welcome / help
    else:
        return prefix + """Hello! I am your **Generative AI Supply Chain Copilot**. I can help you analyze delivery risks and recommend operational interventions.

You can ask me questions like:
- *"Which regions have the highest delay risks?"*
- *"Show me the delay risk distribution across shipping modes"*
- *"What is our total financial revenue at risk?"*
- *"Analyze delay risk for Customer ID 3"*
"""

