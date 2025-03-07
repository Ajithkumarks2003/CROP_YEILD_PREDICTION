import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="Interactive Dashboard", layout="wide")

# Add a title and description
st.title("📊 Interactive Data Dashboard")
st.markdown("This dashboard allows you to explore and visualize your data with various interactive elements.")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Data Explorer", "Time Series Analysis", "Custom Visualization"])

# Sample data generation
@st.cache_data
def generate_sample_data():
    # Create date range for the last 30 days
    dates = [datetime.now() - timedelta(days=x) for x in range(30)]
    dates.reverse()
    
    # Create sample data
    data = {
        'Date': dates,
        'Sales': np.random.normal(100, 15, 30).cumsum(),
        'Visitors': np.random.normal(500, 50, 30).cumsum(),
        'Conversion': np.random.uniform(1, 5, 30),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 30),
        'Product': np.random.choice(['A', 'B', 'C', 'D'], 30)
    }
    return pd.DataFrame(data)

# Generate or load data
with st.sidebar:
    st.header("Dashboard Controls")
    data_option = st.radio("Choose data source:", ["Use sample data", "Upload your CSV"])
    
    if data_option == "Upload your CSV":
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            st.info("Please upload a CSV file or switch to sample data")
            df = generate_sample_data()
    else:
        df = generate_sample_data()
    
    # Global filters
    st.subheader("Filters")
    if 'Region' in df.columns:
        regions = st.multiselect("Select Regions:", options=df['Region'].unique(), default=df['Region'].unique())
        df_filtered = df[df['Region'].isin(regions)]
    else:
        df_filtered = df
    
    # Display dataset info
    st.subheader("Dataset Info")
    st.write(f"Rows: {df_filtered.shape[0]}")
    st.write(f"Columns: {df_filtered.shape[1]}")

# Tab 1: Data Explorer
with tab1:
    st.header("Data Explorer")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Data Preview")
        st.dataframe(df_filtered.head(10), use_container_width=True)
    
    with col2:
        st.subheader("Column Statistics")
        selected_column = st.selectbox("Select a column:", df_filtered.select_dtypes(include=['number']).columns)
        
        stats = {
            "Mean": df_filtered[selected_column].mean(),
            "Median": df_filtered[selected_column].median(),
            "Std Dev": df_filtered[selected_column].std(),
            "Min": df_filtered[selected_column].min(),
            "Max": df_filtered[selected_column].max()
        }
        
        for stat, value in stats.items():
            st.metric(label=stat, value=f"{value:.2f}")
    
    # Data visualization
    st.subheader("Data Visualization")
    viz_type = st.radio("Choose visualization:", ["Bar Chart", "Scatter Plot", "Histogram", "Box Plot"], horizontal=True)
    
    numeric_cols = df_filtered.select_dtypes(include=['number']).columns
    
    if viz_type == "Bar Chart" and len(numeric_cols) > 0:
        x_col = st.selectbox("X-axis (categorical):", df_filtered.columns, key="bar_x")
        y_col = st.selectbox("Y-axis (numeric):", numeric_cols, key="bar_y")
        fig = px.bar(df_filtered, x=x_col, y=y_col, color=x_col)
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Scatter Plot" and len(numeric_cols) >= 2:
        x_col = st.selectbox("X-axis:", numeric_cols, key="scatter_x")
        y_col = st.selectbox("Y-axis:", numeric_cols, key="scatter_y", index=min(1, len(numeric_cols)-1))
        color_col = st.selectbox("Color by:", df_filtered.columns, key="scatter_color")
        fig = px.scatter(df_filtered, x=x_col, y=y_col, color=color_col, size_max=10)
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Histogram" and len(numeric_cols) > 0:
        col = st.selectbox("Column:", numeric_cols, key="hist_col")
        bins = st.slider("Number of bins:", 5, 50, 20)
        fig = px.histogram(df_filtered, x=col, nbins=bins)
        st.plotly_chart(fig, use_container_width=True)
        
    elif viz_type == "Box Plot" and len(numeric_cols) > 0:
        y_col = st.selectbox("Values:", numeric_cols, key="box_y")
        x_col = st.selectbox("Group by:", df_filtered.columns, key="box_x")
        fig = px.box(df_filtered, x=x_col, y=y_col)
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Time Series Analysis
with tab2:
    st.header("Time Series Analysis")
    
    if 'Date' in df_filtered.columns:
        # Ensure date column is properly formatted
        try:
            df_filtered['Date'] = pd.to_datetime(df_filtered['Date'])
            df_filtered = df_filtered.sort_values('Date')
            
            # Select time series data to display
            ts_col = st.selectbox("Select metric to analyze:", 
                                df_filtered.select_dtypes(include=['number']).columns,
                                key="ts_col")
            
            # Time range filter
            date_range = st.date_input(
                "Select date range:",
                [df_filtered['Date'].min().date(), df_filtered['Date'].max().date()],
                min_value=df_filtered['Date'].min().date(),
                max_value=df_filtered['Date'].max().date()
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                mask = (df_filtered['Date'].dt.date >= start_date) & (df_filtered['Date'].dt.date <= end_date)
                filtered_ts = df_filtered.loc[mask]
                
                # Create time series plot
                fig = px.line(filtered_ts, x='Date', y=ts_col, title=f'{ts_col} Over Time')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Add moving average option
                if st.checkbox("Show Moving Average"):
                    window = st.slider("Window size:", 2, 10, 3)
                    filtered_ts[f'{ts_col}_MA'] = filtered_ts[ts_col].rolling(window=window).mean()
                    fig2 = px.line(filtered_ts, x='Date', y=[ts_col, f'{ts_col}_MA'], 
                                    title=f'{ts_col} with {window}-day Moving Average')
                    fig2.update_layout(height=400)
                    st.plotly_chart(fig2, use_container_width=True)
                
                # Additional time series stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean", f"{filtered_ts[ts_col].mean():.2f}")
                with col2:
                    last_val = filtered_ts[ts_col].iloc[-1]
                    prev_val = filtered_ts[ts_col].iloc[-2] if len(filtered_ts) > 1 else last_val
                    change = ((last_val - prev_val) / prev_val * 100) if prev_val != 0 else 0
                    st.metric("Latest Value", f"{last_val:.2f}", f"{change:.1f}%")
                with col3:
                    st.metric("Total", f"{filtered_ts[ts_col].sum():.2f}")

        except Exception as e:
            st.error(f"Error processing time series data: {e}")
    else:
        st.info("No date column found in the dataset. Please upload data with a 'Date' column for time series analysis.")

# Tab 3: Custom Visualization
with tab3:
    st.header("Custom Visualization")
    
    chart_type = st.selectbox("Select chart type:", 
                           ["Heatmap", "Bubble Chart", "Pie Chart", "3D Scatter Plot"])
    
    if chart_type == "Heatmap":
        numeric_cols = df_filtered.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) >= 2:
            st.subheader("Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 8))
            heatmap = sns.heatmap(df_filtered[numeric_cols].corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
            st.pyplot(fig)
        else:
            st.info("Need at least 2 numeric columns for a heatmap")
    
    elif chart_type == "Bubble Chart":
        if len(df_filtered.select_dtypes(include=['number']).columns) >= 3:
            x_col = st.selectbox("X-axis:", df_filtered.select_dtypes(include=['number']).columns, key="bubble_x")
            y_col = st.selectbox("Y-axis:", df_filtered.select_dtypes(include=['number']).columns, key="bubble_y", index=1)
            size_col = st.selectbox("Bubble size:", df_filtered.select_dtypes(include=['number']).columns, key="bubble_size", index=2)
            color_col = st.selectbox("Color:", df_filtered.columns, key="bubble_color")
            
            fig = px.scatter(df_filtered, x=x_col, y=y_col, size=size_col, color=color_col,
                             title=f"Bubble Chart of {x_col} vs {y_col} (size: {size_col})")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 3 numeric columns for a bubble chart")
    
    elif chart_type == "Pie Chart":
        if len(df_filtered.columns) >= 1:
            category_col = st.selectbox("Category:", df_filtered.columns, key="pie_cat")
            value_col = st.selectbox("Values:", df_filtered.select_dtypes(include=['number']).columns, key="pie_val")
            
            # Group by the category and sum the values
            pie_data = df_filtered.groupby(category_col)[value_col].sum().reset_index()
            
            fig = px.pie(pie_data, values=value_col, names=category_col, 
                         title=f"Distribution of {value_col} by {category_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 1 categorical and 1 numeric column for a pie chart")
    
    elif chart_type == "3D Scatter Plot":
        if len(df_filtered.select_dtypes(include=['number']).columns) >= 3:
            x_col = st.selectbox("X-axis:", df_filtered.select_dtypes(include=['number']).columns, key="3d_x")
            y_col = st.selectbox("Y-axis:", df_filtered.select_dtypes(include=['number']).columns, key="3d_y", index=1)
            z_col = st.selectbox("Z-axis:", df_filtered.select_dtypes(include=['number']).columns, key="3d_z", index=2)
            color_col = st.selectbox("Color:", df_filtered.columns, key="3d_color")
            
            fig = px.scatter_3d(df_filtered, x=x_col, y=y_col, z=z_col, color=color_col,
                              title=f"3D Scatter Plot of {x_col}, {y_col}, and {z_col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 3 numeric columns for a 3D scatter plot")

# Add download capability
st.subheader("Download Filtered Data")
csv = df_filtered.to_csv(index=False)
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="filtered_dashboard_data.csv",
    mime="text/csv",
)

# Footer
st.markdown("---")
st.markdown("Created with Streamlit - Interactive Dashboard Template")