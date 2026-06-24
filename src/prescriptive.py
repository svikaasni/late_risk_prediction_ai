import pandas as pd
import numpy as np

def compute_risk_priority_index(prob, complexity_score):
    """
    Risk Priority Index (RPI) = Risk Probability * Order Complexity Score * 100.
    Ranges from 0 to 100.
    """
    return prob * complexity_score * 100

def compute_revenue_at_risk(prob, sales):
    """
    Revenue at Risk (RaR) = Sales * Risk Probability.
    Defines the financial exposure of this order being late.
    """
    return sales * prob

def compute_sla_risk_index(prob, scheduled_days):
    """
    SLA Risk Index = Risk Probability * (5 - scheduled_days) * 20.
    Higher risk when the delivery window is tight (e.g. scheduled days = 1 or 2).
    """
    weight = max(1.0, 5.0 - scheduled_days)
    return prob * weight * 20.0

def compute_intervention_score(prob, complexity_score, sales, scheduled_days):
    """
    Intervention Recommendation Score (IRS) combines RPI and RaR.
    IRS = 0.4 * RPI + 0.6 * min(100.0, RaR / 10)
    Ranges from 0 to 100.
    """
    rpi = compute_risk_priority_index(prob, complexity_score)
    rar = compute_revenue_at_risk(prob, sales)
    normalized_rar = min(100.0, rar / 10.0)  # Caps at $1000 at risk for scaling
    
    return 0.4 * rpi + 0.6 * normalized_rar

def get_prescriptive_recommendations(order_row, prob):
    """
    Generates prescriptive logistics decisions and estimated costs/benefits
    based on the predicted delay risk and features.
    """
    shipping_mode = order_row.get("Shipping Mode", "Standard Class")
    sales = order_row.get("Sales", 0.0)
    market = order_row.get("Market", "LATAM")
    dest_country = order_row.get("Order Country", "United States")
    customer_country = order_row.get("Customer Country", "United States")
    
    recommendations = []
    
    # 1. Shipping Mode Upgrade Rule
    if prob >= 0.70:
        if shipping_mode == "Standard Class":
            cost = max(15.0, sales * 0.05)
            recommendations.append({
                "Action": "Upgrade Shipping Mode to Express (First Class)",
                "Department": "Logistics Dispatch",
                "Cost ($)": round(cost, 2),
                "Risk Reduction (%)": 65.0,
                "Justification": "Standard Class scheduled shipping of 4 days has a high probability of bottlenecking on this lane. Upgrading to 1-day scheduled shipping mitigates 65% of delay probability."
            })
        elif shipping_mode == "Second Class":
            cost = max(25.0, sales * 0.08)
            recommendations.append({
                "Action": "Upgrade Shipping Mode to Express (First Class)",
                "Department": "Logistics Dispatch",
                "Cost ($)": round(cost, 2),
                "Risk Reduction (%)": 50.0,
                "Justification": "Second Class scheduled shipping of 2 days is currently failing SLA. Upgrading to First Class is recommended."
            })
        elif shipping_mode == "First Class":
            cost = max(45.0, sales * 0.12)
            recommendations.append({
                "Action": "Expedite to Same Day Shipping",
                "Department": "Logistics Dispatch",
                "Cost ($)": round(cost, 2),
                "Risk Reduction (%)": 40.0,
                "Justification": "Premium First Class order is experiencing heavy congestion. Expedite processing to Same Day carrier routing."
            })
            
    # 2. Warehousing Prioritization Rule
    if prob >= 0.50:
        # Prioritization cost is operational labor
        cost = 5.0
        recommendations.append({
            "Action": "Apply Priority Pick-and-Pack Flag",
            "Department": "Warehouse Operations",
            "Cost ($)": cost,
            "Risk Reduction (%)": 20.0,
            "Justification": "Order is flagged as medium-to-high risk. Expediting the internal warehouse fulfillment queue offsets transit delay risks."
        })
        
    # 3. International Customs Pre-Clearance Rule
    is_international = customer_country != dest_country
    if is_international and prob >= 0.60:
        cost = 10.0
        recommendations.append({
            "Action": "Initiate Pre-customs Clearance Verification",
            "Department": "Compliance & Customs",
            "Cost ($)": cost,
            "Risk Reduction (%)": 30.0,
            "Justification": "International lane shipment is at risk of customs processing delay. Pre-validate commercial invoice and HTS codes."
        })
        
    # 4. Route Rerouting (For high congestion routes)
    is_latin_america = market == "LATAM"
    if is_latin_america and prob >= 0.75:
        cost = 30.0
        recommendations.append({
            "Action": "Reroute Shipment via Alternative Regional Hub",
            "Department": "Carrier Management",
            "Cost ($)": cost,
            "Risk Reduction (%)": 45.0,
            "Justification": "Historical delay rates show heavy regional congestion. Reroute logistics via secondary hub to avoid bottlenecks."
        })
        
    # 5. Customer Notification (No cost, high CSAT value)
    if prob >= 0.80:
        recommendations.append({
            "Action": "Trigger Proactive Delay Notice to Client Success Team",
            "Department": "Customer Experience",
            "Cost ($)": 0.0,
            "Risk Reduction (%)": 0.0,
            "Justification": "Delay probability exceeds 80%. Notify Account Manager immediately to coordinate customer communication and manage expectations."
        })
        
    # Standard fallback if low risk
    if not recommendations:
        recommendations.append({
            "Action": "Proceed with Standard Shipping",
            "Department": "Logistics Dispatch",
            "Cost ($)": 0.0,
            "Risk Reduction (%)": 0.0,
            "Justification": "Order delay risk is low. Monitor delivery progress in standard cycle."
        })
        
    return recommendations
