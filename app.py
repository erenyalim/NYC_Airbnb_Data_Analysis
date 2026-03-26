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
    df['est_occupancy_days'] = (df['number_of_reviews'] * 2).clip(upper=365)
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
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Filtered Listings", f"{df_filtered.shape[0]:,}")
    m2.metric("Active Hosts", f"{df_filtered['host_id'].nunique():,}")
    m3.metric("Selected Boroughs", f"{df_filtered['neighbourhood_group'].nunique()}")
    m4.metric("Average Price", f"${df_filtered['price'].mean():.2f}")

    st.divider()    

    st.subheader("Dataset Scale")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Rows", f"{df.shape[0]:,}")
    m2.metric("Total Columns", f"{df.shape[1]}")
    m3.metric("Unique Hosts", f"{df['host_id'].nunique():,}")
    m4.metric("Unique Neighbourhoods", f"{df['neighbourhood'].nunique():,}")
    st.divider()


    st.subheader("📊 Dynamic Data Types Analysis")

    custom_features = ['est_occupancy_days', 'est_annual_revenue', 'reviews_per_month']
    type_analysis = pd.DataFrame({
    "Column Name": df_filtered.columns,
    "Data Type": [str(x) for x in df_filtered.dtypes],
    "Non-Null Count": df_filtered.count().values,
    "Unique Values": [df_filtered[col].nunique() for col in df_filtered.columns]
    })

    st.dataframe(type_analysis, use_container_width=True, hide_index=True)

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
    st.title(f"📊 Strategic Insights for {role}")
    st.markdown("---")

    tab_stats, tab_visual, tab_norm = st.tabs(["📊 Statistical Analysis", "🖼️ Graphical Analysis", "⚖️ Data Normalization"])

    with tab_stats:
        st.header(f"📈 {role} Key Metrics")
        meaningful_cols = ['price', 'minimum_nights', 'number_of_reviews', 'availability_365', 'est_annual_revenue']
        selected_cols = st.multiselect("Variables:", options=meaningful_cols, default=['price', 'number_of_reviews', 'est_annual_revenue'])

        if len(selected_cols) > 1:
            c1, c2 = st.columns(2)
            with c1:
                st.write("Central Tendency & Dispersion")
                stats_summary = df_filtered[selected_cols].agg(['mean', 'var', 'std', 'median']).T
                st.dataframe(stats_summary.style.background_gradient(cmap='Blues'))
            with c2:
                st.write("Correlation Matrix")
                fig_corr = px.imshow(df_filtered[selected_cols].corr(), text_auto=".2f", color_continuous_scale='RdBu_r', range_color=[-1,1])
                st.plotly_chart(fig_corr, use_container_width=True)

    with tab_visual:
        st.header(f"🎨 {role} Graphical Breakdown")
        if "Investor" in role:
            fig = px.bar(df_filtered.groupby('neighbourhood_group')['est_annual_revenue'].mean().sort_values(ascending=False).reset_index(), 
                         x='neighbourhood_group', y='est_annual_revenue', color='neighbourhood_group', title="Revenue Potential by Borough")
        elif "Guest" in role:
            fig = px.scatter(df_filtered.sample(min(1500, len(df_filtered))), x='price', y='number_of_reviews', 
                             color='neighbourhood_group', hover_name='name', title="Price-Popularity Trade-off")
        else:
            fig = px.box(df_filtered, x='room_type', y='price', color='room_type', title="Price Distribution by Room Type")
        
        st.plotly_chart(fig, use_container_width=True)

    with tab_norm:
        st.header("⚖️ Data Normalization & Distribution")
        scaler = StandardScaler()
        scaled_prices = scaler.fit_transform(df_filtered[['price']])
        
        cn1, cn2 = st.columns(2)
        with cn1:
            st.write("Original Price Distribution")
            st.plotly_chart(px.histogram(df_filtered, x='price', nbins=50, color_discrete_sequence=['#ff5a5f']), use_container_width=True)
        with cn2:
            st.write("Normalized Z-Score Distribution")
            st.plotly_chart(px.histogram(scaled_prices, nbins=50, color_discrete_sequence=['#00a699']), use_container_width=True)

    st.divider()



