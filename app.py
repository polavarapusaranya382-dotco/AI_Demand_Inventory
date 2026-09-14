import streamlit as st
import pandas as pd
import numpy as np


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Project FORESIGHT",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# BASIC STYLING
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        [data-testid="stMetricValue"] {
            font-size: 28px;
        }

        h1 {
            font-weight: 700;
        }

        h2 {
            font-weight: 650;
        }

        h3 {
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATA LOADERS
# ============================================================

@st.cache_data
def load_sales():

    df = pd.read_csv(
        "data/sales_10k_clean.csv"
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    return df


@st.cache_data
def load_risk():

    df = pd.read_csv(
        "data/final_risk_scoring.csv"
    )

    return df


# ============================================================
# LOAD DATA
# ============================================================

try:

    sales = load_sales()
    risk = load_risk()

except Exception as e:

    st.error(
        "Data files could not be loaded."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# FORECAST PREPARATION
# Cached for performance
# ============================================================

@st.cache_data
def prepare_weekly_forecast(sales):

    weekly = (
        sales
        .groupby("sku_id")
        .resample("W", on="date")
        .agg(
            units_sold=("quantity", "sum"),
            revenue=("total_value", "sum"),
            transactions=("receipt_id", "nunique")
        )
        .reset_index()
    )

    weekly["seasonal_naive_forecast"] = (
        weekly
        .groupby("sku_id")["units_sold"]
        .shift(52)
    )

    return weekly


weekly_forecast = prepare_weekly_forecast(
    sales
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📦 Project FORESIGHT")

st.sidebar.write(
    "AI-Powered Demand & Inventory Intelligence"
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Sales Analytics",
        "📈 Forecasting",
        "📦 Inventory Dashboard",
        "🚨 Risk Dashboard",
        "🔎 Product Details",
        "📋 Executive Summary"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Data scope: 10,000 cleaned sales transactions"
)


# ============================================================
# HOME PAGE
# ============================================================

if page == "🏠 Home":

    st.title(
        "📊 AI-Powered Demand & Inventory Intelligence Platform"
    )

    st.subheader(
        "Project FORESIGHT"
    )

    st.write(
        "Executive overview of sales demand, inventory health "
        "and stock risk for NorthBay Living."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    total_records = len(sales)

    total_units = sales["quantity"].sum()

    total_revenue = sales["total_value"].sum()

    unique_skus = sales["sku_id"].nunique()

    risk_records = len(risk)

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Sales Records",
            f"{total_records:,}"
        )

    with c2:

        st.metric(
            "Units Sold",
            f"{total_units:,.0f}"
        )

    with c3:

        st.metric(
            "Total Revenue",
            f"₹{total_revenue:,.0f}"
        )

    with c4:

        st.metric(
            "Inventory Risk Records",
            f"{risk_records:,}"
        )

    st.markdown("---")

    # --------------------------------------------------------
    # PROJECT SCOPE
    # --------------------------------------------------------

    st.header("🎯 Project Scope")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            ### 📊 Sales Intelligence

            • 10,000 cleaned sales transactions  
            • Demand trend analysis  
            • Product performance  
            • Promotion analysis  
            • Weekly demand patterns  
            """
        )

    with col2:

        st.markdown(
            """
            ### 📦 Inventory Intelligence

            • Inventory risk identification  
            • Stockout detection  
            • Reorder recommendations  
            • Overstock identification  
            • Action-oriented risk classification  
            """
        )

    st.markdown("---")

    # --------------------------------------------------------
    # QUICK DEMAND TREND
    # --------------------------------------------------------

    st.header("📈 Overall Demand Trend")

    daily_home = (
        sales
        .groupby("date")
        .agg(
            units_sold=("quantity", "sum"),
            revenue=("total_value", "sum")
        )
        .sort_index()
    )

    st.line_chart(
        daily_home
    )

    st.markdown("---")

    # --------------------------------------------------------
    # RISK OVERVIEW
    # --------------------------------------------------------

    st.header("🚨 Inventory Risk Overview")

    if "final_risk_category" in risk.columns:

        risk_summary = (
            risk["final_risk_category"]
            .value_counts()
            .reset_index()
        )

        risk_summary.columns = [
            "Risk Category",
            "Records"
        ]

        st.dataframe(
            risk_summary,
            use_container_width=True,
            hide_index=True
        )

    elif "risk_category" in risk.columns:

        risk_summary = (
            risk["risk_category"]
            .value_counts()
            .reset_index()
        )

        risk_summary.columns = [
            "Risk Category",
            "Records"
        ]

        st.dataframe(
            risk_summary,
            use_container_width=True,
            hide_index=True
        )

    st.success(
        "Dashboard is powered by the 10,000 cleaned sales "
        "transactions used throughout Project FORESIGHT."
    )


# ============================================================
# SALES ANALYTICS
# ============================================================

elif page == "📊 Sales Analytics":

    st.title("📊 Sales Analytics")

    st.write(
        "Sales performance and demand analysis based on "
        "10,000 cleaned sales transactions."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    st.subheader("🔎 Filters")

    c1, c2 = st.columns(2)

    with c1:

        stores = (
            ["All Stores"] +
            sorted(
                sales["store_id"]
                .dropna()
                .unique()
                .tolist()
            )
        )

        selected_store = st.selectbox(
            "Select Store",
            stores
        )

    with c2:

        channels = (
            ["All Channels"] +
            sorted(
                sales["channel"]
                .dropna()
                .unique()
                .tolist()
            )
        )

        selected_channel = st.selectbox(
            "Select Channel",
            channels
        )

    filtered = sales.copy()

    if selected_store != "All Stores":

        filtered = filtered[
            filtered["store_id"] == selected_store
        ]

    if selected_channel != "All Channels":

        filtered = filtered[
            filtered["channel"] == selected_channel
        ]

    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("📌 Sales KPIs")

    k1, k2, k3, k4 = st.columns(4)

    with k1:

        st.metric(
            "Transactions",
            f"{filtered['receipt_id'].nunique():,}"
        )

    with k2:

        st.metric(
            "Units Sold",
            f"{filtered['quantity'].sum():,.0f}"
        )

    with k3:

        st.metric(
            "Revenue",
            f"₹{filtered['total_value'].sum():,.0f}"
        )

    with k4:

        st.metric(
            "Unique SKUs",
            f"{filtered['sku_id'].nunique():,}"
        )

    # --------------------------------------------------------
    # DAILY TREND
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("📈 Daily Sales Trend")

    daily = (
        filtered
        .groupby("date")
        .agg(
            units_sold=("quantity", "sum"),
            revenue=("total_value", "sum")
        )
        .sort_index()
    )

    st.line_chart(
        daily
    )

    # --------------------------------------------------------
    # WEEKLY TREND
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("📅 Weekly Demand Trend")

    weekly = (
        filtered
        .set_index("date")
        .resample("W")
        .agg(
            units_sold=("quantity", "sum"),
            revenue=("total_value", "sum")
        )
    )

    st.line_chart(
        weekly
    )

    # --------------------------------------------------------
    # TOP PRODUCTS
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("🏆 Top 10 Products")

    top_products = (
        filtered
        .groupby("sku_id")
        .agg(
            units_sold=("quantity", "sum"),
            revenue=("total_value", "sum"),
            transactions=("receipt_id", "nunique")
        )
        .sort_values(
            "units_sold",
            ascending=False
        )
        .head(10)
        .reset_index()
    )

    st.dataframe(
        top_products,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        top_products.set_index("sku_id")[
            "units_sold"
        ]
    )

    # --------------------------------------------------------
    # PROMOTION
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("🎯 Promotion Analysis")

    promo = filtered.copy()

    promo["is_promotion"] = (
        promo["promo_id"].notna()
    )

    promo_summary = (
        promo
        .groupby("is_promotion")
        .agg(
            transactions=("receipt_id", "nunique"),
            units_sold=("quantity", "sum"),
            revenue=("total_value", "sum"),
            avg_quantity=("quantity", "mean"),
            avg_revenue=("total_value", "mean")
        )
        .reset_index()
    )

    promo_summary["is_promotion"] = (
        promo_summary["is_promotion"]
        .map(
            {
                True: "Promotion",
                False: "No Promotion"
            }
        )
    )

    st.dataframe(
        promo_summary,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # MONTHLY
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("📆 Monthly Sales Performance")

    monthly = (
        filtered
        .set_index("date")
        .resample("ME")
        .agg(
            units_sold=("quantity", "sum"),
            revenue=("total_value", "sum")
        )
    )

    st.line_chart(
        monthly
    )

    # --------------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("📋 Filtered Sales Data")

    st.write(
        f"Showing **{len(filtered):,}** records."
    )

    st.dataframe(
        filtered.head(100),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FORECASTING
# ============================================================

elif page == "📈 Forecasting":

    st.title("📈 Demand Forecast")

    st.write(
        "Weekly SKU-level demand forecast using the selected "
        "Seasonal-Naive forecasting model."
    )

    st.markdown("---")

    forecast_skus = (
        weekly_forecast[
            weekly_forecast[
                "seasonal_naive_forecast"
            ].notna()
        ]["sku_id"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if not forecast_skus:

        st.warning(
            "No SKU has enough historical data for "
            "a 52-week seasonal forecast."
        )

    else:

        selected_sku = st.selectbox(
            "Select SKU",
            forecast_skus
        )

        sku_data = weekly_forecast[
            weekly_forecast["sku_id"] == selected_sku
        ].copy()

        sku_data = sku_data.sort_values(
            "date"
        )

        latest_actual = (
            sku_data.iloc[-1]["units_sold"]
        )

        valid_forecast = sku_data[
            sku_data[
                "seasonal_naive_forecast"
            ].notna()
        ]

        if not valid_forecast.empty:

            latest_forecast = (
                valid_forecast.iloc[-1][
                    "seasonal_naive_forecast"
                ]
            )

        else:

            latest_forecast = 0

        # ----------------------------------------------------
        # KPI
        # ----------------------------------------------------

        st.subheader("📌 Forecast KPIs")

        k1, k2, k3, k4 = st.columns(4)

        with k1:

            st.metric(
                "Selected SKU",
                selected_sku
            )

        with k2:

            st.metric(
                "Latest Actual Demand",
                f"{latest_actual:,.0f}"
            )

        with k3:

            st.metric(
                "Seasonal-Naive Forecast",
                f"{latest_forecast:,.2f}"
            )

        with k4:

            st.metric(
                "Forecast Horizon",
                "6 Weeks"
            )

        # ----------------------------------------------------
        # MODEL PERFORMANCE
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader("📊 Model Performance")

        m1, m2, m3 = st.columns(3)

        with m1:

            st.metric(
                "Selected Model",
                "Seasonal-Naive"
            )

        with m2:

            st.metric(
                "Rolling WAPE",
                "144.27%"
            )

        with m3:

            st.metric(
                "Rolling Bias",
                "-0.038"
            )

        st.info(
            "Seasonal-Naive was selected because it achieved "
            "the best rolling-origin WAPE among the tested "
            "forecasting approaches."
        )

        # ----------------------------------------------------
        # HISTORICAL
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "📈 Historical Weekly Demand"
        )

        history = (
            sku_data[
                ["date", "units_sold"]
            ]
            .tail(52)
            .set_index("date")
        )

        st.line_chart(
            history
        )

        # ----------------------------------------------------
        # ACTUAL VS FORECAST
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "📊 Actual Demand vs Forecast"
        )

        comparison = (
            sku_data[
                [
                    "date",
                    "units_sold",
                    "seasonal_naive_forecast"
                ]
            ]
            .dropna()
            .tail(52)
            .set_index("date")
        )

        st.line_chart(
            comparison
        )

        # ----------------------------------------------------
        # NEXT 6 WEEKS
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "🔮 Next 6 Weeks Forecast"
        )

        last_date = sku_data["date"].max()

        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(weeks=1),
            periods=6,
            freq="W"
        )

        lookup = (
            sku_data
            .set_index("date")["units_sold"]
            .to_dict()
        )

        future_values = []

        for future_date in future_dates:

            source_date = (
                future_date -
                pd.Timedelta(weeks=52)
            )

            value = lookup.get(
                source_date,
                np.nan
            )

            future_values.append(
                value
            )

        future_df = pd.DataFrame(
            {
                "Forecast Week": future_dates,
                "Forecast Demand": future_values
            }
        )

        future_df["Forecast Demand"] = (
            future_df["Forecast Demand"]
            .fillna(0)
            .round(2)
        )

        st.dataframe(
            future_df,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            future_df.set_index(
                "Forecast Week"
            )["Forecast Demand"]
        )

        # ----------------------------------------------------
        # BUSINESS INTERPRETATION
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "💡 Business Interpretation"
        )

        total_forecast = (
            future_df["Forecast Demand"].sum()
        )

        avg_forecast = (
            future_df["Forecast Demand"].mean()
        )

        st.write(
            f"For **{selected_sku}**, estimated demand "
            f"over the next 6 weeks is approximately "
            f"**{total_forecast:,.0f} units**."
        )

        st.write(
            f"Average weekly forecast demand is "
            f"approximately **{avg_forecast:,.2f} units**."
        )


# ============================================================
# INVENTORY DASHBOARD
# ============================================================

elif page == "📦 Inventory Dashboard":

    st.title("📦 Inventory Dashboard")

    st.write(
        "Inventory health, stock position and replenishment "
        "visibility for the planning team."
    )

    st.markdown("---")

    inventory = risk.copy()

    # --------------------------------------------------------
    # KPI
    # --------------------------------------------------------

    stockout_count = 0
    below_safety_count = 0
    reorder_count = 0

    if "stock_on_hand" in inventory.columns:

        stockout_count = (
            inventory["stock_on_hand"] <= 0
        ).sum()

    if (
        "stock_on_hand" in inventory.columns
        and "safety_stock" in inventory.columns
    ):

        below_safety_count = (
            inventory["stock_on_hand"]
            <
            inventory["safety_stock"]
        ).sum()

    if (
        "stock_on_hand" in inventory.columns
        and "reorder_point" in inventory.columns
    ):

        reorder_count = (
            inventory["stock_on_hand"]
            <=
            inventory["reorder_point"]
        ).sum()

    k1, k2, k3, k4 = st.columns(4)

    with k1:

        st.metric(
            "Inventory Records",
            f"{len(inventory):,}"
        )

    with k2:

        st.metric(
            "Stockout Records",
            f"{stockout_count:,}"
        )

    with k3:

        st.metric(
            "Below Safety Stock",
            f"{below_safety_count:,}"
        )

    with k4:

        st.metric(
            "Reorder Point Risk",
            f"{reorder_count:,}"
        )

    # --------------------------------------------------------
    # STOCK POSITION
    # --------------------------------------------------------

    if (
        "stock_on_hand" in inventory.columns
        and "reorder_point" in inventory.columns
    ):

        st.markdown("---")

        st.subheader(
            "📊 Stock Position"
        )

        stock_view = inventory[
            [
                "sku_id",
                "stock_on_hand",
                "reorder_point",
                "safety_stock"
            ]
        ].copy()

        stock_view = (
            stock_view
            .sort_values(
                "stock_on_hand"
            )
            .head(20)
        )

        st.dataframe(
            stock_view,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # INVENTORY HEALTH
    # --------------------------------------------------------

    if "final_risk_category" in inventory.columns:

        st.markdown("---")

        st.subheader(
            "🩺 Inventory Health Distribution"
        )

        health = (
            inventory[
                "final_risk_category"
            ]
            .value_counts()
            .reset_index()
        )

        health.columns = [
            "Risk Category",
            "Records"
        ]

        st.bar_chart(
            health.set_index(
                "Risk Category"
            )
        )

        st.dataframe(
            health,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # INVENTORY TABLE
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "📋 Inventory Planning View"
    )

    display_cols = [
        c for c in [
            "store_id",
            "sku_id",
            "stock_on_hand",
            "reorder_point",
            "safety_stock",
            "final_risk_category",
            "recommended_action"
        ]
        if c in inventory.columns
    ]

    if display_cols:

        st.dataframe(
            inventory[display_cols].head(100),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# RISK DASHBOARD
# ============================================================

elif page == "🚨 Risk Dashboard":

    st.title("🚨 Inventory Risk Dashboard")

    st.write(
        "Action-oriented stockout, replenishment and "
        "overstock risk monitoring."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # RISK CATEGORY
    # --------------------------------------------------------

    if "final_risk_category" in risk.columns:

        risk_category = (
            risk["final_risk_category"]
            .value_counts()
            .reset_index()
        )

        risk_category.columns = [
            "Risk Category",
            "Records"
        ]

        st.subheader(
            "🚨 Risk Category Overview"
        )

        st.dataframe(
            risk_category,
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            risk_category.set_index(
                "Risk Category"
            )
        )

    # --------------------------------------------------------
    # ACTIONS
    # --------------------------------------------------------

    if "recommended_action" in risk.columns:

        st.markdown("---")

        st.subheader(
            "🎯 Recommended Actions"
        )

        action_summary = (
            risk["recommended_action"]
            .value_counts()
            .reset_index()
        )

        action_summary.columns = [
            "Recommended Action",
            "Records"
        ]

        st.dataframe(
            action_summary,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # CRITICAL STOCKOUT
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "🔴 Critical Stockout Records"
    )

    if "final_risk_category" in risk.columns:

        critical = risk[
            risk["final_risk_category"]
            == "Critical Stockout"
        ].copy()

        st.write(
            f"Critical stockout records: "
            f"**{len(critical):,}**"
        )

        critical_cols = [
            c for c in [
                "store_id",
                "sku_id",
                "stock_on_hand",
                "reorder_point",
                "safety_stock",
                "forecast_weekly_demand",
                "final_risk_category",
                "recommended_action"
            ]
            if c in critical.columns
        ]

        if critical_cols:

            st.dataframe(
                critical[critical_cols].head(100),
                use_container_width=True,
                hide_index=True
            )

    # --------------------------------------------------------
    # FILTER RISK
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "🔎 Risk Filter"
    )

    if "final_risk_category" in risk.columns:

        risk_options = (
            ["All Risk Categories"] +
            sorted(
                risk[
                    "final_risk_category"
                ]
                .dropna()
                .unique()
                .tolist()
            )
        )

        selected_risk = st.selectbox(
            "Select Risk Category",
            risk_options
        )

        risk_filtered = risk.copy()

        if selected_risk != "All Risk Categories":

            risk_filtered = risk_filtered[
                risk_filtered[
                    "final_risk_category"
                ] == selected_risk
            ]

        st.write(
            f"Showing **{len(risk_filtered):,}** records."
        )

        st.dataframe(
            risk_filtered.head(200),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PRODUCT DETAILS
# ============================================================

elif page == "🔎 Product Details":

    st.title("🔎 Product Details")

    st.write(
        "Detailed demand, sales and inventory information "
        "for an individual SKU."
    )

    st.markdown("---")

    product_skus = (
        sales["sku_id"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    selected_product = st.selectbox(
        "Select Product / SKU",
        product_skus
    )

    product_sales = sales[
        sales["sku_id"] == selected_product
    ].copy()

    # --------------------------------------------------------
    # SALES KPI
    # --------------------------------------------------------

    st.subheader(
        "📊 Product Sales KPIs"
    )

    total_units = (
        product_sales["quantity"].sum()
    )

    total_revenue = (
        product_sales["total_value"].sum()
    )

    transactions = (
        product_sales["receipt_id"].nunique()
    )

    avg_quantity = (
        product_sales["quantity"].mean()
    )

    k1, k2, k3, k4 = st.columns(4)

    with k1:

        st.metric(
            "Units Sold",
            f"{total_units:,.0f}"
        )

    with k2:

        st.metric(
            "Revenue",
            f"₹{total_revenue:,.0f}"
        )

    with k3:

        st.metric(
            "Transactions",
            f"{transactions:,}"
        )

    with k4:

        st.metric(
            "Avg Quantity",
            f"{avg_quantity:,.2f}"
        )

    # --------------------------------------------------------
    # PRODUCT TREND
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "📈 Product Demand Trend"
    )

    product_daily = (
        product_sales
        .groupby("date")
        ["quantity"]
        .sum()
    )

    st.line_chart(
        product_daily
    )

    # --------------------------------------------------------
    # INVENTORY INFORMATION
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "📦 Inventory Position"
    )

    if "sku_id" in risk.columns:

        product_risk = risk[
            risk["sku_id"] == selected_product
        ].copy()

        if not product_risk.empty:

            inventory_cols = [
                c for c in [
                    "store_id",
                    "sku_id",
                    "stock_on_hand",
                    "reorder_point",
                    "safety_stock",
                    "avg_weekly_demand",
                    "weeks_of_inventory",
                    "forecast_weekly_demand",
                    "forecast_weeks_of_cover",
                    "final_risk_category",
                    "recommended_action"
                ]
                if c in product_risk.columns
            ]

            if inventory_cols:

                st.dataframe(
                    product_risk[
                        inventory_cols
                    ],
                    use_container_width=True,
                    hide_index=True
                )

        else:

            st.info(
                "No inventory record found for this SKU."
            )

    # --------------------------------------------------------
    # PRODUCT FORECAST
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader(
        "🔮 Product Forecast"
    )

    product_forecast = weekly_forecast[
        weekly_forecast["sku_id"]
        == selected_product
    ].copy()

    product_forecast = (
        product_forecast
        .dropna(
            subset=[
                "seasonal_naive_forecast"
            ]
        )
    )

    if not product_forecast.empty:

        forecast_value = (
            product_forecast
            .iloc[-1]
            ["seasonal_naive_forecast"]
        )

        st.metric(
            "Latest Seasonal-Naive Forecast",
            f"{forecast_value:,.2f} units"
        )

        forecast_chart = (
            product_forecast[
                [
                    "date",
                    "units_sold",
                    "seasonal_naive_forecast"
                ]
            ]
            .tail(52)
            .set_index("date")
        )

        st.line_chart(
            forecast_chart
        )

    else:

        st.warning(
            "Seasonal forecast is not available "
            "for this SKU."
        )


# ============================================================
# EXECUTIVE SUMMARY
# ============================================================

elif page == "📋 Executive Summary":

    st.title(
        "📋 Executive Summary Dashboard"
    )

    st.write(
        "Executive-level view of demand, forecasting "
        "performance and inventory risk."
    )

    st.markdown("---")

    # --------------------------------------------------------
    # BUSINESS KPIs
    # --------------------------------------------------------

    st.header(
        "💼 Business KPIs"
    )

    total_units = sales["quantity"].sum()

    total_revenue = sales["total_value"].sum()

    unique_skus = sales["sku_id"].nunique()

    sales_records = len(sales)

    k1, k2, k3, k4 = st.columns(4)

    with k1:

        st.metric(
            "Sales Records",
            f"{sales_records:,}"
        )

    with k2:

        st.metric(
            "Units Sold",
            f"{total_units:,.0f}"
        )

    with k3:

        st.metric(
            "Revenue",
            f"₹{total_revenue:,.0f}"
        )

    with k4:

        st.metric(
            "Unique SKUs",
            f"{unique_skus:,}"
        )

    # --------------------------------------------------------
    # FORECAST PERFORMANCE
    # --------------------------------------------------------

    st.markdown("---")

    st.header(
        "📈 Forecast Performance"
    )

    f1, f2, f3 = st.columns(3)

    with f1:

        st.metric(
            "Selected Model",
            "Seasonal-Naive"
        )

    with f2:

        st.metric(
            "Rolling WAPE",
            "144.27%"
        )

    with f3:

        st.metric(
            "Rolling Bias",
            "-0.038"
        )

    st.info(
        "The Seasonal-Naive model was selected because "
        "it achieved the best rolling-origin WAPE among "
        "the tested models."
    )

    # --------------------------------------------------------
    # RISK SUMMARY
    # --------------------------------------------------------

    st.markdown("---")

    st.header(
        "🚨 Inventory Risk Summary"
    )

    if "final_risk_category" in risk.columns:

        risk_summary = (
            risk[
                "final_risk_category"
            ]
            .value_counts()
            .reset_index()
        )

        risk_summary.columns = [
            "Risk Category",
            "Records"
        ]

        st.dataframe(
            risk_summary,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------
    # RECOMMENDATIONS
    # --------------------------------------------------------

    st.markdown("---")

    st.header(
        "🎯 Business Recommendations"
    )

    st.markdown(
        """
        ### 1. Prioritize Critical Stockouts
        Immediately review SKUs with zero stock and
        prioritize replenishment for high-demand items.

        ### 2. Monitor Reorder Risks
        SKUs approaching or falling below their reorder
        point should be included in the replenishment plan.

        ### 3. Review Overstock Candidates
        Products with excess inventory and weak demand
        should be reviewed for clearance or markdown actions.

        ### 4. Use Forecast for Planning
        The Seasonal-Naive forecast provides a transparent
        baseline for weekly SKU-level demand planning.

        ### 5. Monitor Forecast Performance
        Forecast accuracy should be reviewed regularly using
        rolling-origin WAPE and bias.

        ### 6. Keep the Process Reproducible
        The dashboard should always be refreshed from the
        cleaned analytical dataset rather than manually
        edited numbers.
        """
    )

    # --------------------------------------------------------
    # FINAL MESSAGE
    # --------------------------------------------------------

    st.markdown("---")

    st.success(
        "Project FORESIGHT dashboard is ready for "
        "sales, forecasting, inventory and risk analysis."
    )

    st.caption(
        "All sales analysis is based on exactly "
        "10,000 cleaned sales transactions."
    )