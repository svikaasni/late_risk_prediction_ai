import os
import sys
# Add project root directory to Python path for robust imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
import joblib

# Set Page Config first (must be the very first Streamlit command)
st.set_page_config(
    page_title="APL Logistics Decision Intelligence Platform",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Temporary debugging on Streamlit Cloud (Sidebar)
st.sidebar.markdown("### 🔍 Debug Info")
st.sidebar.write("CWD:", os.getcwd())
try:
    st.sidebar.write("Root files:", os.listdir("."))
    if os.path.exists("models"):
        st.sidebar.write("Models files:", os.listdir("models"))
    else:
        st.sidebar.write("models/ folder missing!")
except Exception as e:
    st.sidebar.write("Error listing files:", e)

# App custom CSS styling ( Sleek dark mode / glassmorphism theme)
st.markdown("""
<style>
    /* Main layout modifications */
    .reportview-container {
        background: #0f172a;
        color: #f8fafc;
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #1e293b;
    }
    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #94a3b8;
        font-weight: 500;
    }
    /* Custom Card container */
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    }
    .high-risk-badge {
        background-color: #ef4444;
        color: white;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: bold;
    }
    .medium-risk-badge {
        background-color: #f59e0b;
        color: white;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: bold;
    }
    .low-risk-badge {
        background-color: #10b981;
        color: white;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Imports from src
from utils import load_ml_pipeline, get_dashboard_data_sample, predict_df_risks, handle_copilot_query
from src.explainability import explain_single_order
from src.prescriptive import get_prescriptive_recommendations, compute_intervention_score

# Load model, pipeline and sample data
model, pipeline, explainer, shap_sample = load_ml_pipeline()
data_sample = None

if model is None or pipeline is None:
    st.error("⚠️ Machine Learning pipeline artifacts not found. Please run the model training pipeline first!")
    st.info("To run the training pipeline, run: `python -m src.train` in your terminal.")
else:
    data_sample = get_dashboard_data_sample()
    # Batch predict risks for the sample
    data_sample = predict_df_risks(data_sample, model, pipeline)

# Navigation
st.sidebar.title("🚚 APL Logistics")
st.sidebar.subheader("Decision Intelligence Platform")

page = st.sidebar.radio(
    "Navigation Menu",
    [
        "Executive Overview", 
        "Operations Control Tower", 
        "Order Risk Prediction", 
        "What-If Simulator", 
        "Shipping Analytics", 
        "Global Risk Map",
        "Generative AI Copilot"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**ML Model Status:** `ACTIVE (LightGBM)`")
st.sidebar.markdown(f"**Data Version:** `2026.06.24` ({len(data_sample) if data_sample is not None else 0} pending orders)")

if data_sample is not None:
    # ----------------------------------------------------
    # PAGE 1: EXECUTIVE OVERVIEW
    # ----------------------------------------------------
    if page == "Executive Overview":
        st.title("📊 Executive Logistics Risk Dashboard")
        st.subheader("High-level KPIs, delay trends, and revenue exposure")
        
        # Calculations
        total_orders = len(data_sample)
        delay_rate = (data_sample["Risk_Level"] != "Low").mean()
        total_rar = data_sample["Revenue_at_Risk"].sum()
        avg_risk = data_sample["Risk_Probability"].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"<div class='metric-card'><h3>📦 Total Orders</h3><h2>{total_orders:,}</h2><p style='color:#94a3b8;'>Active shipping batch</p></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><h3>⚠️ Delay Risk Rate</h3><h2>{delay_rate*100:.1f}%</h2><p style='color:#ef4444;'>Medium & High risk orders</p></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><h3>💵 Revenue at Risk</h3><h2>${total_rar:,.2f}</h2><p style='color:#f59e0b;'>Weighted delay exposure</p></div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div class='metric-card'><h3>🎯 Avg Delay Prob.</h3><h2>{avg_risk*100:.1f}%</h2><p style='color:#10b981;'>Mean model probability</p></div>", unsafe_allow_html=True)
            
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("### 📈 Predicted SLA Compliance Trends (Nov - Dec 2025)")
            # Resample by date
            daily_stats = data_sample.groupby(data_sample["Order Date"].dt.date).agg(
                Total_Orders=("Sales", "count"),
                Avg_Risk=("Risk_Probability", "mean")
            ).reset_index()
            daily_stats["SLA_Compliance"] = (1.0 - daily_stats["Avg_Risk"]) * 100
            
            fig = px.line(
                daily_stats, x="Order Date", y="SLA_Compliance",
                title="Predicted SLA Compliance Trend (Target: 95%)",
                labels={"SLA_Compliance": "SLA Compliance (%)", "Order Date": "Order Date"},
                template="plotly_dark", color_discrete_sequence=["#38bdf8"]
            )
            # Add target line
            fig.add_hline(y=95.0, line_dash="dash", line_color="#ef4444", annotation_text="95% Target SLA")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.markdown("### 🎯 Order Risk Category Distribution")
            risk_dist = data_sample["Risk_Level"].value_counts().reset_index()
            # Ensure compatibility with both pandas 1.x and 2.x column naming
            risk_dist.columns = ["Risk_Level", "count"]
            
            fig_pie = px.pie(
                risk_dist, names="Risk_Level", values="count",
                color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
                hole=0.4, template="plotly_dark"
            )
            fig_pie.update_layout(height=400, margin=dict(t=30, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        # Regional Exposure Table
        st.markdown("### 🌍 Top Regional Financial Exposures")
        reg_exposure = data_sample.groupby("Order Region").agg(
            Orders=("Sales", "count"),
            Total_Sales=("Sales", "sum"),
            Revenue_at_Risk=("Revenue_at_Risk", "sum"),
            Avg_Delay_Risk=("Risk_Probability", "mean")
        ).reset_index().sort_values(by="Revenue_at_Risk", ascending=False)
        
        reg_exposure["Avg_Delay_Risk"] = reg_exposure["Avg_Delay_Risk"].apply(lambda x: f"{x*100:.1f}%")
        reg_exposure["Total_Sales"] = reg_exposure["Total_Sales"].apply(lambda x: f"${x:,.2f}")
        reg_exposure["Revenue_at_Risk"] = reg_exposure["Revenue_at_Risk"].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(reg_exposure, use_container_width=True)

    # ----------------------------------------------------
    # PAGE 2: OPERATIONS CONTROL TOWER (Digital Twin)
    # ----------------------------------------------------
    elif page == "Operations Control Tower":
        st.title("🎛️ Operations Control Tower")
        st.subheader("Active shipping queue prioritized by Risk Priority Index (RPI) and Revenue at Risk (RaR)")
        
        st.markdown("This queue filters pending orders with **Medium & High** delay risk. Operations managers can select an order, simulate upgrading the shipping mode, and immediately observe risk mitigation.")
        
        # Filter high/medium risk
        risk_queue = data_sample[data_sample["Risk_Level"].isin(["Medium", "High"])].copy()
        risk_queue = risk_queue.sort_values(by="Intervention_Score", ascending=False).reset_index(drop=True)
        
        if risk_queue.empty:
            st.success("✅ Excellent! No active orders are currently in Medium/High risk.")
        else:
            # Table visualization
            display_cols = [
                "Customer Id", "Shipping Mode", "Market", "Order Region", "Product Name", 
                "Sales", "Risk_Probability", "Risk_Level", "Intervention_Score"
            ]
            
            # Format table for display
            display_df = risk_queue[display_cols].copy()
            display_df["Sales"] = display_df["Sales"].apply(lambda x: f"${x:,.2f}")
            display_df["Risk_Probability"] = display_df["Risk_Probability"].apply(lambda x: f"{x*100:.1f}%")
            display_df["Intervention_Score"] = display_df["Intervention_Score"].round(1)
            
            st.markdown(f"**Showing {len(display_df)} High/Medium Risk Orders**")
            
            # Select Order to Triage
            selected_idx = st.selectbox("Select Customer ID to Triage / Simulate Intervention:", display_df.index, 
                                        format_func=lambda idx: f"Cust ID: {display_df.loc[idx, 'Customer Id']} | Product: {display_df.loc[idx, 'Product Name']} | Risk: {display_df.loc[idx, 'Risk_Probability']} | IRS: {display_df.loc[idx, 'Intervention_Score']}")
            
            order_data = risk_queue.iloc[selected_idx].to_dict()
            prob = order_data["Risk_Probability"]
            
            # Draw Order Details card
            st.markdown("---")
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("### 📋 Active Order Profile")
                risk_badge = f"<span class='high-risk-badge'>HIGH</span>" if order_data["Risk_Level"] == "High" else f"<span class='medium-risk-badge'>MEDIUM</span>"
                st.markdown(f"**Risk Level**: {risk_badge}", unsafe_allow_html=True)
                st.markdown(f"**Customer ID**: `{order_data['Customer Id']}`")
                st.markdown(f"**Product**: `{order_data['Product Name']}`")
                st.markdown(f"**Shipping Mode**: `{order_data['Shipping Mode']}`")
                st.markdown(f"**Destination**: `{order_data['Order City']}, {order_data['Order Country']}`")
                st.markdown(f"**Sales Value**: `${order_data['Sales']:,.2f}`")
                st.markdown(f"**Revenue at Risk**: `${order_data['Revenue_at_Risk']:,.2f}`")
                st.markdown(f"**Intervention Score**: `{order_data['Intervention_Score']:.1f} / 100`")
                
            with c2:
                st.markdown("### 🛠️ Prescriptive Recommendation Engine")
                recs = get_prescriptive_recommendations(order_data, prob)
                
                # Show in a table
                rec_df = pd.DataFrame(recs)
                st.dataframe(rec_df, use_container_width=True)
                
                # Action Simulator Button
                st.markdown("#### ⚡ Digital Twin Simulation")
                has_upgrade = any("Upgrade" in r["Action"] for r in recs)
                
                if has_upgrade:
                    if st.button("🚀 Simulate Express Shipping Mode Upgrade"):
                        # Simulate upgrade: change shipping mode to First Class
                        simulated_order = order_data.copy()
                        simulated_order["Shipping Mode"] = "First Class"
                        simulated_order["Days for shipment (scheduled)"] = 1.0
                        
                        # Re-run prediction
                        explanation_res = explain_single_order(simulated_order, model, pipeline, explainer)
                        new_prob = explanation_res["probability"]
                        new_level = explanation_res["risk_level"]
                        
                        new_badge = f"<span class='low-risk-badge'>LOW</span>" if new_level == "Low" else f"<span class='medium-risk-badge'>MEDIUM</span>"
                        st.balloons()
                        st.success("🎉 Simulation Successful!")
                        st.markdown(f"**New Predicted Risk**: {new_badge} (**{new_prob*100:.1f}%** delay probability)", unsafe_allow_html=True)
                        st.markdown(f"**Risk reduction**: **-{((prob - new_prob)*100):.1f}%** absolute decrease!")
                        st.info("Action: Shipping upgrade order dispatched to carrier. SLA compliance safeguarded.")
                else:
                    st.write("Order is already routed via Express / Same Day or cannot be upgraded further.")
                    
            st.markdown("### 🗂️ Active Risk Queue Details")
            st.dataframe(display_df, use_container_width=True)

    # ----------------------------------------------------
    # PAGE 3: ORDER RISK PREDICTION & SHAP EXPLAINER
    # ----------------------------------------------------
    elif page == "Order Risk Prediction":
        st.title("🔮 Predictive Risk Engine")
        st.subheader("Score single shipments and retrieve explainable AI diagnostics")
        
        st.markdown("Input the parameters of an incoming order below to calculate its late delivery risk *before* dispatching to the warehouse.")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🖊️ Order Attributes")
            # Select boxes populated with unique values from sample
            shipping_mode_input = st.selectbox("Shipping Mode", ["Standard Class", "Second Class", "First Class", "Same Day"])
            market_input = st.selectbox("Market", ["LATAM", "Europe", "Pacific Asia", "USCA", "Africa"])
            order_region_input = st.selectbox("Order Region", sorted(data_sample["Order Region"].unique()))
            order_country_input = st.selectbox("Order Country", sorted(data_sample["Order Country"].unique()))
            order_city_input = st.selectbox("Order City", sorted(data_sample["Order City"].unique()))
            
            category_name_input = st.selectbox("Product Category", sorted(data_sample["Category Name"].unique()))
            product_name_input = st.selectbox("Product Name", sorted(data_sample["Product Name"].unique()))
            
            type_input = st.selectbox("Payment Type", ["DEBIT", "TRANSFER", "CASH", "PAYMENT"])
            segment_input = st.selectbox("Customer Segment", ["Consumer", "Corporate", "Home Office"])
            
        with c2:
            st.markdown("### 💰 Financials & Routing")
            sales_input = st.slider("Gross Sales ($)", 10.0, 2000.0, 250.0)
            quantity_input = st.slider("Quantity", 1, 10, 2)
            discount_rate_input = st.slider("Discount Rate", 0.0, 0.25, 0.05, step=0.01)
            
            benefit_input = st.slider("Expected Profit ($)", -500.0, 1000.0, 50.0)
            latitude_input = st.number_input("Customer Latitude", value=18.0)
            longitude_input = st.number_input("Customer Longitude", value=-66.0)
            
            # Map scheduled days based on Shipping Mode (deterministic in dataset)
            mode_scheduled_mapping = {
                "Standard Class": 4.0,
                "Second Class": 2.0,
                "First Class": 1.0,
                "Same Day": 0.0
            }
            scheduled_days = mode_scheduled_mapping[shipping_mode_input]
            
            # Prepare row dictionary
            input_row = {
                "Type": type_input,
                "Days for shipment (scheduled)": scheduled_days,
                "Benefit per order": benefit_input,
                "Sales per customer": sales_input * (1 - discount_rate_input), # Net sales
                "Category Name": category_name_input,
                "Customer City": "Caguas", # Default placeholder city
                "Customer Country": "Puerto Rico",
                "Customer Id": 9999, # Dummy ID
                "Customer Segment": segment_input,
                "Customer State": "PR",
                "Department Name": "Fan Shop",
                "Latitude": latitude_input,
                "Longitude": longitude_input,
                "Market": market_input,
                "Order City": order_city_input,
                "Order Country": order_country_input,
                "Order Item Discount": sales_input * discount_rate_input,
                "Order Item Discount Rate": discount_rate_input,
                "Order Item Product Price": sales_input / quantity_input,
                "Order Item Profit Ratio": benefit_input / (sales_input + 0.1),
                "Order Item Quantity": quantity_input,
                "Sales": sales_input,
                "Order Item Total": sales_input * (1 - discount_rate_input),
                "Order Profit Per Order": benefit_input,
                "Order Region": order_region_input,
                "Order State": "San Juan",
                "Product Name": product_name_input,
                "Shipping Mode": shipping_mode_input
            }
            
            # Predict Button
            st.markdown("---")
            if st.button("🔮 Calculate Delay Risk", use_container_width=True):
                if explainer is None:
                    st.error("SHAP Explainer not available.")
                else:
                    # Run explanation
                    res = explain_single_order(input_row, model, pipeline, explainer)
                    prob = res["probability"]
                    
                    st.markdown("---")
                    st.markdown("### 🏆 Prediction Results")
                    
                    col_p1, col_p2 = st.columns([1, 2])
                    with col_p1:
                        # Color based on risk level
                        badge_color = "#ef4444" if res["risk_level"] == "High" else "#f59e0b" if res["risk_level"] == "Medium" else "#10b981"
                        st.markdown(f"""
                        <div style='background-color:{badge_color}; padding:20px; border-radius:12px; text-align:center;'>
                            <h3 style='color:white; margin:0;'>{res['risk_level'].upper()} RISK</h3>
                            <h1 style='color:white; margin:10px 0;'>{prob*100:.1f}%</h1>
                            <p style='color:white; margin:0;'>Delay Probability</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col_p2:
                        st.markdown(res["explanation"])
                        
                    # SHAP waterfall plot
                    st.markdown("### 📊 Local SHAP Waterfall Plot (Why is it risky?)")
                    fig_shap, ax_shap = plt.subplots(figsize=(10, 4))
                    # Plot waterfall
                    shap.plots.waterfall(res["shap_values"], max_display=10, show=False)
                    plt.tight_layout()
                    st.pyplot(fig_shap)
                    plt.close(fig_shap)

    # ----------------------------------------------------
    # PAGE 4: WHAT-IF SIMULATOR
    # ----------------------------------------------------
    elif page == "What-If Simulator":
        st.title("🎛️ What-If Risk Simulator")
        st.subheader("Simulate order modifications in real time to optimize logistics risk")
        
        st.markdown("Use the controls below to change order parameters and see how the delay risk shifts dynamically. This allows you to perform cost-vs-risk optimization before booking.")
        
        # Load a default row from the test set
        base_order = data_sample.iloc[0].to_dict()
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("### ⚙️ Simulator Controls")
            sim_shipping_mode = st.selectbox("Select Shipping Mode:", ["Standard Class", "Second Class", "First Class", "Same Day"], index=0)
            sim_region = st.selectbox("Destination Region:", sorted(data_sample["Order Region"].unique()), index=0)
            sim_qty = st.slider("Quantity:", 1, 10, int(base_order.get("Order Item Quantity", 1)))
            sim_discount = st.slider("Discount Rate:", 0.0, 0.25, float(base_order.get("Order Item Discount Rate", 0.05)), step=0.01)
            
            # Map scheduled days based on Shipping Mode
            mode_scheduled_mapping = {
                "Standard Class": 4.0,
                "Second Class": 2.0,
                "First Class": 1.0,
                "Same Day": 0.0
            }
            sim_scheduled_days = mode_scheduled_mapping[sim_shipping_mode]
            
        with c2:
            # Construct simulated row
            sim_row = base_order.copy()
            sim_row["Shipping Mode"] = sim_shipping_mode
            sim_row["Days for shipment (scheduled)"] = sim_scheduled_days
            sim_row["Order Region"] = sim_region
            sim_row["Order Item Quantity"] = sim_qty
            sim_row["Order Item Discount Rate"] = sim_discount
            # Adjust dependent calculations
            sim_row["Order Item Discount"] = sim_row["Sales"] * sim_discount
            sim_row["Order Item Total"] = sim_row["Sales"] * (1 - sim_discount)
            sim_row["Sales per customer"] = sim_row["Sales"] * (1 - sim_discount)
            
            # Predict
            fe = pipeline["feature_engineer"]
            preprocessor = pipeline["preprocessor"]
            
            df_sim_raw = pd.DataFrame([sim_row])
            df_sim_fe = fe.transform(df_sim_raw)
            
            from src.config import COLS_TO_DROP
            cols_to_drop = [c for c in COLS_TO_DROP if c in df_sim_fe.columns]
            X_sim_raw = df_sim_fe.drop(columns=cols_to_drop + ["Late_delivery_risk", "Order Date", "Risk_Probability", "Risk_Level", "Risk_Priority_Index", "Revenue_at_Risk", "SLA_Risk_Index", "Intervention_Score"], errors="ignore")
            
            X_sim_processed = preprocessor.transform(X_sim_raw)
            sim_prob = model.predict_proba(X_sim_processed)[0, 1]
            
            # Visualize gauge or bar
            st.markdown("### 🏆 Real-Time Simulated Risk")
            
            # Color
            sim_color = "#ef4444" if sim_prob >= 0.70 else "#f59e0b" if sim_prob >= 0.30 else "#10b981"
            sim_level = "High" if sim_prob >= 0.70 else "Medium" if sim_prob >= 0.30 else "Low"
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = sim_prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Simulated Risk: {sim_level.upper()}"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': sim_color},
                    'steps' : [
                        {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.15)"},
                        {'range': [30, 70], 'color': "rgba(245, 158, 11, 0.15)"},
                        {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.15)"}
                    ],
                }
            ))
            fig.update_layout(template="plotly_dark", height=250, margin=dict(t=50, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)
            
            # Write a recommendation notice based on simulated risk
            if sim_level == "High":
                st.error("⚠️ Warning: Simulated configuration yields high risk of delivery delay. We strongly recommend reducing quantities, upgrading shipping mode, or routing through regional hubs.")
            elif sim_level == "Medium":
                st.warning("⚠️ Caution: Simulated configuration has moderate delay risk. Monitor scheduled departure closely.")
            else:
                st.success("✅ Success: Simulated configuration is low risk. Safe to proceed with these logistics parameters.")

    # ----------------------------------------------------
    # PAGE 5: SHIPPING ANALYTICS
    # ----------------------------------------------------
    elif page == "Shipping Analytics":
        st.title("📈 Shipping Performance Analytics")
        st.subheader("Performance drilldown across different categories, carriers, and markets")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🏷️ Delay Risk by Shipping Mode")
            mode_chart_data = data_sample.groupby("Shipping Mode")["Risk_Probability"].mean().reset_index()
            mode_chart_data["Delay_Risk_%"] = mode_chart_data["Risk_Probability"] * 100
            
            fig = px.bar(
                mode_chart_data, x="Shipping Mode", y="Delay_Risk_%",
                color="Shipping Mode", title="Average Delay Risk per Shipping Mode",
                template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.markdown("### 📊 Delay Risk by Market")
            market_chart_data = data_sample.groupby("Market")["Risk_Probability"].mean().reset_index()
            market_chart_data["Delay_Risk_%"] = market_chart_data["Risk_Probability"] * 100
            
            fig = px.bar(
                market_chart_data, x="Market", y="Delay_Risk_%",
                color="Market", title="Average Delay Risk per Market",
                template="plotly_dark", color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("### 🛍️ Top 10 Product Categories by Delay Risk")
        cat_chart_data = data_sample.groupby("Category Name")["Risk_Probability"].agg(["mean", "count"]).reset_index()
        # Filter for categories with significant order counts (e.g. count > 5)
        cat_chart_data = cat_chart_data[cat_chart_data["count"] >= 5]
        cat_chart_data = cat_chart_data.sort_values(by="mean", ascending=False).head(10)
        cat_chart_data["Delay_Risk_%"] = cat_chart_data["mean"] * 100
        
        fig = px.bar(
            cat_chart_data, y="Category Name", x="Delay_Risk_%",
            orientation="h", title="Top 10 Product Categories by Average Delay Risk",
            template="plotly_dark", color="Delay_Risk_%", color_continuous_scale="Reds"
        )
        fig.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------
    # PAGE 6: GLOBAL RISK MAP
    # ----------------------------------------------------
    elif page == "Global Risk Map":
        st.title("🌍 Global Logistics Risk Map")
        st.subheader("Geographical distribution of delay rates and volumes")
        
        st.markdown("This map shows the location of customers (stores) and color-codes them by their average predicted delay risk. Darker red coordinates indicate high delay risks, while green coordinates indicate on-time lanes.")
        
        # Plotly map using customer coordinates
        # Map center coordinates
        map_data = data_sample.copy()
        map_data["Delay_Probability_%"] = map_data["Risk_Probability"] * 100
        
        fig = px.scatter_mapbox(
            map_data, lat="Latitude", lon="Longitude",
            color="Delay_Probability_%", size="Sales",
            color_continuous_scale="RdYlGn_r", size_max=15,
            zoom=2, mapbox_style="carto-darkmatter",
            hover_name="Order City", hover_data=["Order Country", "Product Name", "Sales", "Shipping Mode"],
            title="Global Shipment Risks Map (Size = Sales, Color = Delay Risk)",
            template="plotly_dark"
        )
        fig.update_layout(height=600, margin=dict(t=50, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------
    # PAGE 7: GENERATIVE AI COPILOT
    # ----------------------------------------------------
    elif page == "Generative AI Copilot":
        st.title("💬 Generative AI Supply Chain Copilot")
        st.subheader("Ask questions in natural language and receive intelligence from data and model diagnostics")
        
        st.markdown("Our GenAI Copilot acts as a digital agent. It is fully connected to the active logistics database and model metrics, allowing you to ask queries and immediately get diagnostic recommendations.")
        
        # Session state for chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I am your AI Supply Chain Copilot. Ask me anything about regional congestion, high-risk shipping lanes, financial exposures, or specific order triage plans!"}
            ]
            
        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
        # Handle user input using text_input for 100% layout compatibility
        with st.form("copilot_form", clear_on_submit=True):
            prompt = st.text_input("Type your question below and press Enter or Send:")
            submit = st.form_submit_button("Send 🚀")
            
        if submit and prompt:
            # Append user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate response
            response = handle_copilot_query(prompt, data_sample, model, pipeline)
            
            # Append assistant response
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Force refresh to render new messages in chronological order
            st.rerun()
