import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="NYC Airbnb Analysis", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('AB_NYC_2019.csv')
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    df['est_occupancy_days'] = (df['number_of_reviews'] * 2 * 3).clip(upper=255)    
    df['est_annual_revenue'] = df['est_occupancy_days'] * df['price']
    return df

df = load_data()

role = st.sidebar.selectbox(
    "Select Perspective:", 
    options=[None, "📈 Investor (ROI)", "🏠 Host (Competition)", "🧳 Guest (Value)"],
    format_func=lambda x: "📊 General Dataset Information" if x is None else x
)

st.sidebar.divider()
selected_borough = st.sidebar.multiselect("District Filter:", options=df['neighbourhood_group'].unique(), default=df['neighbourhood_group'].unique())
price_range = st.sidebar.slider("Price Range ($):", 0, 1000, (0, 1000))

df_filtered = df[(df['neighbourhood_group'].isin(selected_borough)) & 
                 (df['price'].between(price_range[0], price_range[1]))] 

if role is None:
    st.title("🏙️ NYC Airbnb Open Data: General Dataset Information")
    
    st.markdown("---")
    col_info_left, col_info_right = st.columns(2, gap="large")
    
    with col_info_left:
        st.markdown("### 📖 Dataset Context & Domain")
        st.info(f"""
        **Domain:** Hospitality & Urban Tourism  
        **Context:** Analyzing **{df.shape[0]:,}** Airbnb listings in New York City (2019). 
        This dataset provides a snapshot of the short-term rental market, crucial for urban 
        planning and business intelligence.
        """)
    
    with col_info_right:
        st.markdown("### 🏠 Core Entities")
        st.success(f"""
        * **Listings:** {df.shape[0]:,} rental units.
        * **Hosts:** {df['host_id'].nunique():,} unique providers.
        * **Areas:** 5 Boroughs & {df['neighbourhood'].nunique()} Neighborhoods.
        * **Engagement:** {df['number_of_reviews'].sum():,} total reviews.
        """)
    st.divider()

    st.subheader('Filtered ')

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Filtered Listings", f"{df_filtered.shape[0]:,}")
    m2.metric("Active Hosts", f"{df_filtered['host_id'].nunique():,}")
    m3.metric("Selected Boroughs", f"{df_filtered['neighbourhood_group'].nunique()}")
    m4.metric("Average Price", f"${df_filtered['price'].mean():.2f}")

    st.divider()    

    st.subheader("📊 Dataset Scale ")
    
    m1, m2, m3, m4, m5 = st.columns(5)
    
    m1.metric("Total Rows", f"{df.shape[0]:,}")
    m2.metric("Original Features", "16")
    m3.metric("Engineered Features", "2", delta="Added", delta_color="normal")
    m4.metric("Unique Hosts", f"{df['host_id'].nunique():,}")
    m5.metric("Neighborhoods", f"{df['neighbourhood'].nunique():,}")

    st.divider()

    engineered_list = ['est_occupancy_days', 'est_annual_revenue']
    original_list = [c for c in df_filtered.columns if c not in engineered_list]

    col_ctrl1, col_ctrl2 = st.columns(2, gap="large")

    with col_ctrl1:
        st.markdown("Dataset Features")
        selected_orig = st.multiselect(
            "Filter Columns:", 
            options=original_list,
            default=original_list,
            key="orig_selector"
        )
        if selected_orig:
            df_orig_analysis = pd.DataFrame({
                "Feature": selected_orig,
                "Data Type": [str(df_filtered[c].dtype) for c in selected_orig],
                "Unique Count": [df_filtered[c].nunique() for c in selected_orig],
                "Null Values": [df_filtered[c].isnull().sum() for c in selected_orig]
            })
            st.dataframe(df_orig_analysis, use_container_width=True, hide_index=True)
        else:
            st.info("Please select at least one.")

    with col_ctrl2:
        st.markdown("Added Features")
        selected_eng = st.multiselect(
            "Filter Columns:", 
            options=engineered_list, 
            default=engineered_list,
            key="eng_selector"
        )
        if selected_eng:
            df_eng_analysis = pd.DataFrame({
                "New Feature": selected_eng,
                "Data Type": [str(df_filtered[c].dtype) for c in selected_eng],
                "Unique Count": [df_filtered[c].nunique() for c in selected_eng],
                "Logic Origin": ["Calculated Logic" for _ in selected_eng]
            })
            st.dataframe(df_eng_analysis, use_container_width=True, hide_index=True)
        else:
            st.warning("Please select at least one.")

    st.subheader("📍 Geographic Distribution of Listings")
    map_data = df_filtered[['latitude', 'longitude', 'neighbourhood_group', 'room_type', 'price', 'name']].dropna()
    sample_size = min(3000, len(map_data))

    if sample_size > 0:
        fig_map = px.scatter_mapbox(
            map_data.sample(sample_size,random_state=42),
            lat="latitude", 
            lon="longitude", 
            color="neighbourhood_group", 
            size="price", 
            size_max=12,
            color_discrete_sequence=px.colors.qualitative.Bold,
            hover_name="name",
            hover_data={
                "latitude": False,
                "longitude": False,
                "price" : "$.2f",
                "room_type": True,
                "neighbourhood_group": True,
            },
            zoom=10,
            center = {"lat": 40.7128, "lon": -74.0060},
            mapbox_style="carto-darkmatter",
            title=f"Sample of {sample_size} Listings Distribution",
            height=600
        )

        fig_map.update_layout(
        margin={"r":0,"t":50,"l":0,"b":0},
        mapbox=dict(
            center={"lat": 40.7128, "lon": -74.0060},
            zoom=10,
            style="carto-darkmatter"
        ),
        legend=dict(title="Boroughs", yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
        
        st.plotly_chart(fig_map, use_container_width=True, config={'scrollZoom': True})
        st.caption("🔍 **Pro-Tip:** Use your mouse wheel to zoom. Larger bubbles indicate higher priced listings. Legend allows filtering boroughs directly on the map.")
    else:
        st.warning("No geographic data available for the selected filters.")    
    st.info("💡 **Tip:** Use the filters in the sidebar to see how the dataset characteristics change across different districts and price points.")

else:
    st.title(f"Perspective : {role}")
    st.markdown("---")

    tab_stats, tab_visual, tab_norm = st.tabs(["Statistical Analysis", "Graphical Analysis", "Data Normalization"])

    if "Investor" in role:
        analysis_cols = ['price', 'est_annual_revenue', 'availability_365', 'reviews_per_month']
     
        st.divider()
        with tab_stats:
            st.subheader("Strategic ROI & Performance Indicators")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Avg Annual Revenue", f"${df_filtered['est_annual_revenue'].mean():,.2f}")
            kpi2.metric("Market Covariance (P vs R)", f"{df_filtered['price'].cov(df_filtered['est_annual_revenue']):.2f}")
            kpi3.metric("Highest Earning Area", df_filtered.groupby('neighbourhood_group')['est_annual_revenue'].mean().idxmax())
            kpi4.metric("Avg Yearly Occupancy", f"{int(df_filtered['est_occupancy_days'].mean())} Days")
            st.divider()
            st.header(f"{role}: Financial Risk & Return Analysis")
            st.write("""
            Does a higher price guarantee higher income? We use Covariance and Central Tendency to measure investment reliability.
            """)
            
            c1, c2 = st.columns([1, 1.2])
            
            with c1:
                st.markdown("#### 📊 Descriptive Statistics")
                stats_summary = df_filtered[analysis_cols].agg(['mean', 'var', 'std', 'median']).T
                formatted_stats = stats_summary.style.format("{:,.2f}").background_gradient(cmap='Greens')
                st.dataframe(formatted_stats, use_container_width=True)
                st.info("""
                **How to read this table:**
                * **Mean:** Your expected average annual target ($).
                * **Std (Risk):** How much the actual income varies. High Std = High Uncertainty.
                * **Median:** The 'Real' middle point. If Median is much lower than Mean, a few luxury homes are skewing the average.
                """)

            with c2:
                st.markdown("#### 🤝 Covariance Matrix: Price vs. Profit")
                cov_matrix = df_filtered[analysis_cols].cov()
                
                fig_cov = px.imshow(
                    cov_matrix, 
                    text_auto=".2f", 
                    color_continuous_scale='Greens',
                    labels=dict(color="Covariance Value"),
                    title="Variable Co-movement Analysis"
                )
                st.plotly_chart(fig_cov, use_container_width=True)
                
                st.success("💡 **Investor Insight:** High positive covariance between Price & Revenue confirms that higher-tier investments yield higher returns in this selection.")

        with tab_visual:
            st.header("🖼️ Market Discovery & Visual Intelligence")
            st.write("Visualizing risk-reward ratios and market saturation for strategic decision making.")
            row1_1, row1_2 = st.columns(2)
            
            with row1_1:
                st.markdown("##### 1. Revenue Potential by Borough")
                df_bar = df_filtered.groupby('neighbourhood_group')['est_annual_revenue'].mean().sort_values(ascending=False).reset_index()
                fig_bar = px.bar(
                    df_bar, 
                    x='neighbourhood_group', 
                    y='est_annual_revenue', 
                    color='est_annual_revenue',
                    color_continuous_scale='Greens',
                    labels={'est_annual_revenue': 'Avg Annual Rev ($)', 'neighbourhood_group': 'Borough'},
                    template="plotly_dark"
                )
                fig_bar.update_layout(showlegend=False, margin=dict(t=30, b=0, l=0, r=0))
                st.plotly_chart(fig_bar, use_container_width=True)
                st.info("💡 **Insight:** Manhattan leads in gross revenue, but watch the growth in Brooklyn for higher margins.")

            with row1_2:
                st.markdown("##### 2. Price Elasticity & Revenue Trends")
                df_sample = df_filtered.sample(min(1200, len(df_filtered)), random_state=42)
                fig_scatter = px.scatter(
                    df_sample, 
                    x='price', 
                    y='est_annual_revenue', 
                    size='reviews_per_month', 
                    color='neighbourhood_group', 
                    trendline="ols",
                    opacity=0.6,
                    hover_name='name',
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
                fig_scatter.update_layout(margin=dict(t=30, b=0, l=0, r=0), legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.caption("🔍 Bubble size represents demand speed (Reviews per month).")
            st.divider()
            row2_1, row2_2 = st.columns(2)
            with row2_1:
                st.markdown("##### 3. Market Saturation (Host Density)")
                fig_tree = px.treemap(
                    df_filtered, 
                    path=[px.Constant("NYC Market"), 'neighbourhood_group', 'room_type'], 
                    values='price', 
                    color='price', 
                    color_continuous_scale='RdYlGn_r', 
                    template="plotly_dark"
                )
                fig_tree.update_layout(margin=dict(t=30, b=0, l=0, r=0))
                st.plotly_chart(fig_tree, use_container_width=True)
                st.info("💡 **Strategy:** Look for smaller rectangles in high-revenue areas to find 'underserved' niches.")

            with row2_2:
                st.markdown("##### 4. Revenue Consistency (Risk Analysis)")
                fig_box = px.box(
                    df_filtered, 
                    x='room_type', 
                    y='est_annual_revenue', 
                    color='room_type',
                    notched=True, 
                    template="plotly_dark",
                    points="outliers" 
                )
                fig_box.update_layout(margin=dict(t=30, b=0, l=0, r=0), showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)
                st.success("🎯 **Risk Tip:** Narrower boxes indicate stable income; long whiskers represent high-risk/high-reward properties.")

        with tab_norm:
            st.header("⚖️ Portfolio Normalization")
            scaler = StandardScaler()
            df_filtered['norm_rev'] = scaler.fit_transform(df_filtered[['est_annual_revenue']])
            cn1, cn2 = st.columns(2)
            with cn1:
                st.write("**Raw Revenue Distribution**")
                st.plotly_chart(px.histogram(df_filtered, x='est_annual_revenue', nbins=50, color_discrete_sequence=['#2ecc71']), use_container_width=True)
            with cn2:
                st.write("**Standardized Revenue (Z-Score)**")
                st.plotly_chart(px.histogram(df_filtered, x='norm_rev', nbins=50, color_discrete_sequence=['#27ae60']), use_container_width=True)

    elif "Host" in role:
        analysis_cols = ['price', 'availability_365', 'calculated_host_listings_count']
        
    elif "Guest" in role:
        analysis_cols = ['price', 'number_of_reviews', 'reviews_per_month']

