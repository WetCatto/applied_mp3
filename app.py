import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Global CO2 Emissions Analysis",
    page_icon="🌍",
    layout="wide"
)

# Title and description
st.title("🌍 Vehicle CO2 Emissions Analysis Dashboard")
st.markdown("""
**Dataset**: [CO2 Emission by Vehicles (Kaggle)](https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles)  
Analyze vehicle CO2 emissions based on engine size, fuel consumption, and vehicle characteristics.
""")

# Load data
@st.cache_data
def load_data():
    try:
        # Try to read the CSV file
        # Expected filename: CO2 Emissions_Canada.csv or similar
        df = pd.read_csv('CO2 Emissions_Canada.csv')
        
        # Clean column names (remove spaces, make consistent)
        df.columns = df.columns.str.strip()
        
        # Common column name variations in this dataset
        column_mapping = {
            'Make': 'Make',
            'Model': 'Model',
            'Vehicle Class': 'Vehicle_Class',
            'Engine Size(L)': 'Engine_Size',
            'Cylinders': 'Cylinders',
            'Transmission': 'Transmission',
            'Fuel Type': 'Fuel_Type',
            'Fuel Consumption City (L/100 km)': 'Fuel_Consumption_City',
            'Fuel Consumption Hwy (L/100 km)': 'Fuel_Consumption_Hwy',
            'Fuel Consumption Comb (L/100 km)': 'Fuel_Consumption_Comb',
            'Fuel Consumption Comb (mpg)': 'Fuel_Consumption_Comb_mpg',
            'CO2 Emissions(g/km)': 'CO2_Emissions'
        }
        
        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        return df
    
    except FileNotFoundError:
        st.error("""
        ❌ **CSV file not found!**
        
        Please download the dataset and place it in the same directory as this app:
        
        1. Download from: https://www.kaggle.com/datasets/debajyotipodder/co2-emission-by-vehicles
        2. Save as: `CO2 Emissions_Canada.csv`
        3. Place it in the same folder as `app.py`
        4. Refresh this page
        """)
        st.stop()
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please check that the CSV file format matches the expected structure.")
        st.stop()

df = load_data()

# Display dataset info
with st.expander("ℹ️ Dataset Information"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Total Makes", df['Make'].nunique())
    
    st.write("**Column Names:**", ', '.join(df.columns.tolist()))
    st.write("**Sample Data:**")
    st.dataframe(df.head())

# Sidebar - Filters
st.sidebar.header("🔍 Filters")

# Make filter
makes_list = sorted(df['Make'].unique())
selected_makes = st.sidebar.multiselect(
    "Select Vehicle Make(s)",
    options=makes_list,
    default=makes_list[:5] if len(makes_list) > 5 else makes_list
)

# Vehicle class filter
classes_list = sorted(df['Vehicle_Class'].unique())
selected_classes = st.sidebar.multiselect(
    "Select Vehicle Class(es)",
    options=classes_list,
    default=classes_list
)

# Engine size range
engine_min = float(df['Engine_Size'].min())
engine_max = float(df['Engine_Size'].max())
engine_range = st.sidebar.slider(
    "Engine Size (L)",
    engine_min,
    engine_max,
    (engine_min, engine_max)
)

# CO2 emissions range
co2_min = int(df['CO2_Emissions'].min())
co2_max = int(df['CO2_Emissions'].max())
co2_range = st.sidebar.slider(
    "CO2 Emissions (g/km)",
    co2_min,
    co2_max,
    (co2_min, co2_max)
)

# Fuel type filter
fuel_types = df['Fuel_Type'].unique().tolist()
selected_fuel = st.sidebar.multiselect(
    "Select Fuel Type(s)",
    options=fuel_types,
    default=fuel_types
)

# Apply filters
filtered_df = df[
    (df['Make'].isin(selected_makes)) &
    (df['Vehicle_Class'].isin(selected_classes)) &
    (df['Engine_Size'] >= engine_range[0]) &
    (df['Engine_Size'] <= engine_range[1]) &
    (df['CO2_Emissions'] >= co2_range[0]) &
    (df['CO2_Emissions'] <= co2_range[1]) &
    (df['Fuel_Type'].isin(selected_fuel))
].copy()

# Main content
st.sidebar.markdown(f"**Filtered Records:** {len(filtered_df)} / {len(df)}")

if len(filtered_df) == 0:
    st.warning("⚠️ No data matches the selected filters. Please adjust your filter criteria.")
    st.stop()

# Key Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Avg CO2 Emissions",
        f"{filtered_df['CO2_Emissions'].mean():.0f} g/km",
        f"{filtered_df['CO2_Emissions'].mean() - df['CO2_Emissions'].mean():.0f}"
    )

with col2:
    st.metric(
        "Avg Engine Size",
        f"{filtered_df['Engine_Size'].mean():.2f} L",
        f"{filtered_df['Engine_Size'].mean() - df['Engine_Size'].mean():.2f}"
    )

with col3:
    st.metric(
        "Avg Fuel Consumption",
        f"{filtered_df['Fuel_Consumption_Comb'].mean():.1f} L/100km",
        f"{filtered_df['Fuel_Consumption_Comb'].mean() - df['Fuel_Consumption_Comb'].mean():.1f}"
    )

with col4:
    st.metric(
        "Vehicle Count",
        len(filtered_df),
        f"{len(filtered_df) - len(df)}"
    )

# Tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Trends", "🔬 Correlations", "🤖 ML Clustering", "📋 Data Table"])

with tab1:
    st.header("Summary Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CO2 Emissions by Make (top 10)
        top_makes = filtered_df['Make'].value_counts().head(10).index
        top_makes_df = filtered_df[filtered_df['Make'].isin(top_makes)]
        
        fig1 = px.box(
            top_makes_df,
            x='Make',
            y='CO2_Emissions',
            title='CO2 Emissions Distribution by Vehicle Make (Top 10)',
            color='Make',
            labels={'CO2_Emissions': 'CO2 Emissions (g/km)'}
        )
        fig1.update_layout(showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # CO2 Emissions by Vehicle Class
        avg_by_class = filtered_df.groupby('Vehicle_Class')['CO2_Emissions'].mean().sort_values(ascending=False).head(15)
        fig2 = px.bar(
            x=avg_by_class.values,
            y=avg_by_class.index,
            orientation='h',
            title='Average CO2 Emissions by Vehicle Class',
            labels={'x': 'Avg CO2 Emissions (g/km)', 'y': 'Vehicle Class'},
            color=avg_by_class.values,
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Fuel Type Distribution
    col3, col4 = st.columns(2)
    
    with col3:
        fuel_counts = filtered_df['Fuel_Type'].value_counts()
        fig3 = px.pie(
            values=fuel_counts.values,
            names=fuel_counts.index,
            title='Fuel Type Distribution',
            hole=0.4
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col4:
        # Cylinders vs CO2
        if 'Cylinders' in filtered_df.columns:
            avg_by_cyl = filtered_df.groupby('Cylinders')['CO2_Emissions'].mean().sort_index()
            fig4 = px.line(
                x=avg_by_cyl.index,
                y=avg_by_cyl.values,
                title='Average CO2 Emissions by Number of Cylinders',
                labels={'x': 'Number of Cylinders', 'y': 'Avg CO2 Emissions (g/km)'},
                markers=True
            )
            st.plotly_chart(fig4, use_container_width=True)

with tab2:
    st.header("Emission Trends Analysis")
    
    # Engine Size vs CO2 Emissions
    sample_size = min(1000, len(filtered_df))
    sample_df = filtered_df.sample(n=sample_size, random_state=42) if len(filtered_df) > sample_size else filtered_df
    
    fig5 = px.scatter(
        sample_df,
        x='Engine_Size',
        y='CO2_Emissions',
        color='Vehicle_Class',
        size='Cylinders' if 'Cylinders' in sample_df.columns else None,
        title=f'Engine Size vs CO2 Emissions (Sample: {sample_size} vehicles)',
        labels={'Engine_Size': 'Engine Size (L)', 'CO2_Emissions': 'CO2 Emissions (g/km)'},
        hover_data=['Make', 'Model']
    )
    st.plotly_chart(fig5, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fuel Consumption City vs Highway
        fig6 = px.scatter(
            sample_df,
            x='Fuel_Consumption_City',
            y='Fuel_Consumption_Hwy',
            color='CO2_Emissions',
            title='City vs Highway Fuel Consumption',
            labels={
                'Fuel_Consumption_City': 'City (L/100km)',
                'Fuel_Consumption_Hwy': 'Highway (L/100km)'
            },
            color_continuous_scale='Viridis',
            hover_data=['Make', 'Model']
        )
        st.plotly_chart(fig6, use_container_width=True)
    
    with col2:
        # Combined Fuel Consumption vs CO2
        fig7 = px.scatter(
            sample_df,
            x='Fuel_Consumption_Comb',
            y='CO2_Emissions',
            color='Fuel_Type',
            title='Combined Fuel Consumption vs CO2 Emissions',
            labels={
                'Fuel_Consumption_Comb': 'Combined Fuel Consumption (L/100km)',
                'CO2_Emissions': 'CO2 Emissions (g/km)'
            },
            hover_data=['Make', 'Model']
        )
        st.plotly_chart(fig7, use_container_width=True)

with tab3:
    st.header("Correlation Analysis")
    
    # Correlation heatmap
    numeric_cols = ['Engine_Size', 'Cylinders', 'Fuel_Consumption_City', 
                    'Fuel_Consumption_Hwy', 'Fuel_Consumption_Comb', 'CO2_Emissions']
    
    # Only include columns that exist in the dataframe
    available_numeric_cols = [col for col in numeric_cols if col in filtered_df.columns]
    
    corr_matrix = filtered_df[available_numeric_cols].corr()
    
    fig8 = px.imshow(
        corr_matrix,
        labels=dict(color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        title='Feature Correlation Heatmap',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        text_auto='.2f'
    )
    st.plotly_chart(fig8, use_container_width=True)
    
    st.subheader("Key Insights")
    
    # Find strongest correlations with CO2
    co2_corr = corr_matrix['CO2_Emissions'].drop('CO2_Emissions').sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Strongest Positive Correlations with CO2:**")
        for feature, corr in co2_corr.head(5).items():
            st.write(f"• {feature}: {corr:.3f}")
    
    with col2:
        st.markdown("**Statistical Summary:**")
        st.write(f"• Mean CO2: {filtered_df['CO2_Emissions'].mean():.1f} g/km")
        st.write(f"• Median CO2: {filtered_df['CO2_Emissions'].median():.1f} g/km")
        st.write(f"• Std Dev: {filtered_df['CO2_Emissions'].std():.1f} g/km")
        st.write(f"• Min CO2: {filtered_df['CO2_Emissions'].min():.1f} g/km")
        st.write(f"• Max CO2: {filtered_df['CO2_Emissions'].max():.1f} g/km")

with tab4:
    st.header("Machine Learning: K-Means Clustering")
    
    st.markdown("""
    This section uses K-Means clustering to group vehicles based on their characteristics.
    Vehicles in the same cluster share similar engine size, fuel consumption, and emission patterns.
    """)
    
    # Number of clusters selector
    n_clusters = st.slider("Select Number of Clusters", 2, 8, 4)
    
    # Prepare data for clustering
    clustering_features = ['Engine_Size', 'Fuel_Consumption_Comb', 'CO2_Emissions']
    if 'Cylinders' in filtered_df.columns:
        clustering_features.append('Cylinders')
    
    # Use a sample for clustering if dataset is large
    clustering_sample_size = min(5000, len(filtered_df))
    clustering_df = filtered_df.sample(n=clustering_sample_size, random_state=42) if len(filtered_df) > clustering_sample_size else filtered_df.copy()
    
    X = clustering_df[clustering_features].copy()
    
    # Handle any missing values
    X = X.dropna()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clustering_df.loc[X.index, 'Cluster'] = kmeans.fit_predict(X_scaled)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 3D scatter plot
        fig9 = px.scatter_3d(
            clustering_df,
            x='Engine_Size',
            y='Fuel_Consumption_Comb',
            z='CO2_Emissions',
            color='Cluster',
            title='Vehicle Clusters (3D View)',
            labels={
                'Engine_Size': 'Engine Size (L)',
                'Fuel_Consumption_Comb': 'Fuel Consumption (L/100km)',
                'CO2_Emissions': 'CO2 Emissions (g/km)'
            },
            hover_data=['Make', 'Vehicle_Class']
        )
        st.plotly_chart(fig9, use_container_width=True)
    
    with col2:
        # 2D scatter plot
        fig10 = px.scatter(
            clustering_df,
            x='Engine_Size',
            y='CO2_Emissions',
            color='Cluster',
            title='Vehicle Clusters (2D View)',
            labels={
                'Engine_Size': 'Engine Size (L)',
                'CO2_Emissions': 'CO2 Emissions (g/km)'
            },
            hover_data=['Make', 'Vehicle_Class', 'Fuel_Consumption_Comb']
        )
        st.plotly_chart(fig10, use_container_width=True)
    
    # Cluster statistics
    st.subheader("Cluster Characteristics")
    
    cluster_stats = clustering_df.groupby('Cluster')[clustering_features].mean().round(2)
    cluster_stats['Count'] = clustering_df.groupby('Cluster').size()
    cluster_stats = cluster_stats.sort_values('CO2_Emissions')
    
    st.dataframe(cluster_stats, use_container_width=True)
    
    # Cluster interpretation
    st.markdown("**Cluster Interpretation:**")
    for i in cluster_stats.index:
        cluster_data = cluster_stats.loc[i]
        efficiency_label = "Low Emissions" if cluster_data['CO2_Emissions'] < filtered_df['CO2_Emissions'].mean() else "High Emissions"
        st.write(f"""
        **Cluster {i}** ({efficiency_label}): {cluster_data['Count']:.0f} vehicles
        - Avg Engine: {cluster_data['Engine_Size']:.2f}L | 
        Fuel: {cluster_data['Fuel_Consumption_Comb']:.1f} L/100km | 
        CO2: {cluster_data['CO2_Emissions']:.0f} g/km
        """)

with tab5:
    st.header("Data Table")
    
    st.markdown(f"**Showing {len(filtered_df)} records**")
    
    # Search functionality
    search_term = st.text_input("🔍 Search by Make or Model", "")
    
    if search_term:
        search_df = filtered_df[
            filtered_df['Make'].str.contains(search_term, case=False, na=False) |
            filtered_df['Model'].str.contains(search_term, case=False, na=False)
        ]
        st.markdown(f"*Found {len(search_df)} matching records*")
        display_df = search_df
    else:
        display_df = filtered_df
    
    # Display options
    show_all = st.checkbox("Show all columns")
    
    if show_all:
        st.dataframe(display_df, use_container_width=True, height=400)
    else:
        display_cols = ['Make', 'Model', 'Vehicle_Class', 'Engine_Size', 
                       'Cylinders', 'Fuel_Type', 'Fuel_Consumption_Comb', 'CO2_Emissions']
        # Only show columns that exist
        display_cols = [col for col in display_cols if col in display_df.columns]
        st.dataframe(display_df[display_cols], use_container_width=True, height=400)
    
    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_co2_emissions.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
**About this Dashboard:**
- **Dataset**: CO2 Emission by Vehicles from Kaggle
- **Features**: Interactive filters, visualizations, correlation analysis, and ML clustering
- **Built with**: Streamlit, Plotly, Scikit-learn
""")

# Instructions in sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📖 How to Use")
    st.markdown("""
    1. **Filter Data**: Use the filters above to narrow down vehicles
    2. **Explore Tabs**: 
        - Overview: Summary statistics
        - Trends: Emission patterns
        - Correlations: Feature relationships
        - ML Clustering: Vehicle grouping
        - Data Table: Raw data view & search
    3. **Download**: Export filtered data from the Data Table tab
    """)