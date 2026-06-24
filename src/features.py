import numpy as np
import pandas as pd
from src.config import TARGET_COL

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Computes the great-circle distance between two points on the Earth's surface in km.
    """
    # Earth's radius in km
    r = 6371.0
    
    # Convert degrees to radians
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    
    return r * c

class LogisticsFeatureEngineer:
    def __init__(self):
        # Dictionary to store historical statistics computed on training set
        self.regional_congestion = {}
        self.route_congestion = {}
        self.customer_reliability = {}
        self.shipping_efficiency_gap = {}
        self.global_delay_mean = 0.5  # default fallback
        self.global_gap_mean = 0.0    # default fallback
        
        # Approximate lat/lon coordinates for the 23 Order Regions
        self.region_coordinates = {
            "South Asia": (20.0, 77.0),
            "Central America": (12.0, -86.0),
            "East of USA": (40.0, -75.0),
            "Southern Europe": (40.0, 15.0),
            "Western Europe": (48.0, 2.0),
            "Oceania": (-25.0, 135.0),
            "South America": (-15.0, -60.0),
            "West Asia": (32.0, 45.0),
            "West Africa": (12.0, -3.0),
            "Caribbean": (21.0, -78.0),
            "US Center": (39.0, -98.0),
            "Central Africa": (0.0, 20.0),
            "Southeast Asia": (10.0, 110.0),
            "Northern Europe": (60.0, 15.0),
            "Eastern Asia": (35.0, 105.0),
            "South of  USA": (33.0, -85.0),
            "Southern Africa": (-28.0, 24.0),
            "West of USA": (37.0, -120.0),
            "Eastern Europe": (50.0, 30.0),
            "Canada": (56.0, -106.0),
            "East Africa": (-1.0, 37.0),
            "North Africa": (26.0, 17.0),
            "Central Asia": (45.0, 65.0)
        }

    def fit(self, train_df):
        """
        Fits feature mappings (e.g. historical statistics) based ONLY on training data.
        This prevents data leakage from the validation/test set.
        """
        print("Fitting logistics feature mappings on training data...")
        self.global_delay_mean = train_df[TARGET_COL].mean()
        
        # 1. Regional Congestion Index (Historical delay rate by region)
        self.regional_congestion = train_df.groupby("Order Region")[TARGET_COL].mean().to_dict()
        
        # 2. Route Congestion Index (Historical delay rate by Customer State -> Order Country)
        self.route_congestion = train_df.groupby(["Customer State", "Order Country"])[TARGET_COL].mean().to_dict()
        
        # 3. Customer Reliability Score (Historical delay rate of Customer)
        self.customer_reliability = train_df.groupby("Customer Id")[TARGET_COL].mean().to_dict()
        
        # 4. Historical Shipping Efficiency Gap (actual_shipping_days - scheduled_shipping_days)
        # Note: actual_shipping_days is 'Days for shipping (real)'
        if "Days for shipping (real)" in train_df.columns:
            gap = train_df["Days for shipping (real)"] - train_df["Days for shipment (scheduled)"]
            self.global_gap_mean = gap.mean()
            self.shipping_efficiency_gap = train_df.groupby(["Shipping Mode", "Order Region"]).apply(
                lambda x: (x["Days for shipping (real)"] - x["Days for shipment (scheduled)"]).mean(),
                include_groups=False
            ).to_dict()
        
        return self

    def transform(self, df):
        """
        Transforms the dataframe, adding engineered features.
        """
        print("Transforming dataset and engineering features...")
        res = df.copy()
        
        # 1. Shipping Pressure Index
        # scheduled_days / quantity (plus 1 to avoid division by zero if quantity is 0)
        res["Shipping Pressure Index"] = res["Days for shipment (scheduled)"] / (res["Order Item Quantity"] + 0.1)
        
        # 2. Order Grouping & Complexity Score
        # We group items into simulated orders using Customer Id, Order City, and Shipping Mode
        order_grouped = res.groupby(["Customer Id", "Order City", "Shipping Mode"])
        
        res["Order_Total_Qty"] = order_grouped["Order Item Quantity"].transform("sum")
        res["Order_Product_Count"] = order_grouped["Product Name"].transform("nunique")
        res["Order_Total_Value"] = order_grouped["Order Item Total"].transform("sum")
        
        # Max values for normalization
        max_qty = res["Order_Total_Qty"].max() if res["Order_Total_Qty"].max() > 0 else 1.0
        max_prod = res["Order_Product_Count"].max() if res["Order_Product_Count"].max() > 0 else 1.0
        max_val = res["Order_Total_Value"].max() if res["Order_Total_Value"].max() > 0 else 1.0
        
        # Weighted combination for Order Complexity Score
        res["Order Complexity Score"] = (
            0.3 * (res["Order_Total_Qty"] / max_qty) + 
            0.3 * (res["Order_Product_Count"] / max_prod) + 
            0.4 * (res["Order_Total_Value"] / max_val)
        )
        
        # 3. Discount Risk Score
        res["Discount Risk Score"] = res["Order Item Discount Rate"] * res["Order Item Quantity"]
        
        # 4. High Value Order Flag
        sales_90 = res["Sales"].quantile(0.90)
        res["High Value Order Flag"] = (res["Sales"] > sales_90).astype(int)
        
        # 5. Express Shipment Flag
        res["Express Shipment Flag"] = res["Shipping Mode"].isin(["First Class", "Same Day"]).astype(int)
        
        # 6. International Shipment Flag
        # If customer and order countries are different
        res["International Shipment Flag"] = (res["Customer Country"] != res["Order Country"]).astype(int)
        
        # 7. Geographical Distance Features
        # Compute great-circle distance between customer lat/lon and the approximate center of the Order Region
        def get_region_coords(region):
            return self.region_coordinates.get(region, (30.0, 0.0))  # Default center if region not found
            
        region_coords = res["Order Region"].apply(get_region_coords)
        res["Region_Lat"] = [c[0] for c in region_coords]
        res["Region_Lon"] = [c[1] for c in region_coords]
        
        res["Geographical Distance"] = haversine_distance(
            res["Latitude"], res["Longitude"], 
            res["Region_Lat"], res["Region_Lon"]
        )
        
        # Drop temporary coordinate columns
        res = res.drop(columns=["Region_Lat", "Region_Lon"])
        
        # 8. Historical Congestion and Reliability Features (using mapped statistics)
        # Regional Congestion Index
        res["Regional Congestion Index"] = res["Order Region"].map(self.regional_congestion).fillna(self.global_delay_mean)
        
        # Route Congestion Index
        route_keys = list(zip(res["Customer State"], res["Order Country"]))
        res["Route Congestion Index"] = [self.route_congestion.get(k, self.global_delay_mean) for k in route_keys]
        
        # Customer Reliability Score
        res["Customer Reliability Score"] = res["Customer Id"].map(self.customer_reliability).fillna(self.global_delay_mean)
        
        # Historical Shipping Efficiency Gap (average gap for Shipping Mode + Order Region)
        ship_region_keys = list(zip(res["Shipping Mode"], res["Order Region"]))
        res["Historical Shipping Efficiency Gap"] = [
            self.shipping_efficiency_gap.get(k, self.global_gap_mean) for k in ship_region_keys
        ]
        
        # Clean up temporary grouping columns to avoid cluttering features
        res = res.drop(columns=["Order_Total_Qty", "Order_Product_Count", "Order_Total_Value"])
        
        return res
