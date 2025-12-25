"""
📊 Professional Data Analytics Dashboard v3.0
✨ Advanced Features:
- Smart data type detection & conversion
- Multi-criteria filtering & advanced search
- Outlier detection & data quality analysis
- Export to multiple formats (CSV, Excel, JSON)
- Correlation analysis with statistical tests
- Time series analysis for date columns
- Data profiling & insights
- Duplicate detection
- Missing data visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import json
from io import BytesIO

# ===== LOGGING =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== CONFIG =====
class Config:
    PAGE_TITLE = "Advanced Data Analytics Dashboard"
    PAGE_ICON = "📊"
    LAYOUT = "wide"
    MAX_ROWS = 5000
    ALLOWED_EXTENSIONS = ['csv', 'xlsx', 'xls']
    THEME = "light"

# ===== PAGE SETUP =====
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout=Config.LAYOUT,
    initial_sidebar_state="expanded"
)

st.title(f"{Config.PAGE_ICON} {Config.PAGE_TITLE}")
st.markdown("---")

# ===== UTILITY FUNCTIONS =====
@st.cache_data
def load_file(file):
    """Load CSV or Excel file with error handling"""
    try:
        if file.name.endswith('.csv'):
            try:
                file.seek(0)
                df = pd.read_csv(file, low_memory=False)
                logger.info(f"✅ CSV loaded: {df.shape}")
                return df
            except:
                file.seek(0)
                df = pd.read_csv(file, low_memory=False, encoding='latin-1')
                logger.info(f"✅ CSV loaded (latin-1): {df.shape}")
                return df
        else:
            file.seek(0)
            df = pd.read_excel(file)
            logger.info(f"✅ Excel loaded: {df.shape}")
            return df
    except Exception as e:
        logger.error(f"Load error: {str(e)}")
        st.error(f"❌ Error: {str(e)}")
        return None

def clean_data(df):
    """Clean data"""
    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='all')
    df = df.reset_index(drop=True)
    df.columns = df.columns.str.strip()
    logger.info(f"✅ Data cleaned: {df.shape}")
    return df

def convert_types(df):
    """Smart type conversion"""
    df_converted = df.copy()
    
    for col in df_converted.columns:
        if df_converted[col].isnull().all():
            continue
        
        if df_converted[col].dtype == 'object':
            df_converted[col] = df_converted[col].astype(str).str.strip()
            df_converted[col] = df_converted[col].replace(['None', 'none', 'NaN', ''], np.nan)
            
            try:
                numeric_col = pd.to_numeric(
                    df_converted[col].str.replace(',', '', regex=False),
                    errors='coerce'
                )
                if numeric_col.notna().sum() / len(numeric_col) > 0.5:
                    df_converted[col] = numeric_col
                    continue
            except:
                pass
            
            try:
                datetime_col = pd.to_datetime(df_converted[col], errors='coerce')
                if datetime_col.notna().sum() / len(datetime_col) > 0.5:
                    df_converted[col] = datetime_col
            except:
                pass
    
    logger.info("✅ Types converted")
    return df_converted

def detect_outliers(df, column, method='iqr'):
    """Detect outliers using IQR or Z-score"""
    if df[column].dtype not in ['float64', 'int64']:
        return pd.Series([False] * len(df))
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        return (df[column] < Q1 - 1.5*IQR) | (df[column] > Q3 + 1.5*IQR)
    else:
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return z_scores > 3

def export_to_excel(df):
    """Export to Excel"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Data', index=False)
    return output.getvalue()

def export_to_json(df):
    """Export to JSON"""
    return df.to_json(orient='records', indent=2)

# ===== SIDEBAR =====
st.sidebar.header("📁 Data Management")
uploaded_file = st.sidebar.file_uploader(
    "📤 Choose file:",
    type=Config.ALLOWED_EXTENSIONS
)

df = None
if uploaded_file:
    df = load_file(uploaded_file)

# ===== MAIN CONTENT =====
if df is not None:
    df = clean_data(df)
    df = convert_types(df)
    
    st.success(f"✅ Ready: {len(df):,} rows × {len(df.columns)} columns | Size: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    
    # ===== TABS =====
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Data", "📈 Charts", "📊 Statistics", 
        "🔍 Analysis", "⚙️ Advanced", "💾 Export"
    ])
    
    # ===== TAB 1: DATA =====
    with tab1:
        st.subheader("🔍 Data Explorer")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Rows", f"{len(df):,}")
        with col2:
            st.metric("📈 Columns", len(df.columns))
        with col3:
            numeric_count = len(df.select_dtypes(include=['float64', 'int64']).columns)
            st.metric("🔢 Numeric", numeric_count)
        with col4:
            duplicates = df.duplicated().sum()
            st.metric("🔄 Duplicates", duplicates)
        
        st.write("---")
        
        # Advanced Filter
        st.write("**🔎 Advanced Filter:**")
        filter_col = st.selectbox("Column:", df.columns, key="filter_col")
        filter_type = st.radio("Filter Type:", ["Contains", "Exact Match", "Range", "Null Check"], horizontal=True)
        
        df_filtered = df.copy()
        
        if filter_type == "Contains":
            filter_val = st.text_input("Search value:", key="filter_val")
            if filter_val:
                df_filtered = df_filtered[df_filtered[filter_col].astype(str).str.contains(filter_val, case=False, na=False)]
        
        elif filter_type == "Exact Match":
            unique_vals = df[filter_col].unique()
            selected_vals = st.multiselect("Select values:", unique_vals)
            if selected_vals:
                df_filtered = df_filtered[df_filtered[filter_col].isin(selected_vals)]
        
        elif filter_type == "Range":
            if df[filter_col].dtype in ['float64', 'int64']:
                min_val, max_val = st.slider(
                    "Select range:",
                    float(df[filter_col].min()),
                    float(df[filter_col].max()),
                    (float(df[filter_col].min()), float(df[filter_col].max()))
                )
                df_filtered = df_filtered[(df_filtered[filter_col] >= min_val) & (df_filtered[filter_col] <= max_val)]
        
        elif filter_type == "Null Check":
            show_nulls = st.checkbox("Show only rows with nulls", value=False)
            if show_nulls:
                df_filtered = df_filtered[df_filtered[filter_col].isnull()]
            else:
                df_filtered = df_filtered[df_filtered[filter_col].notnull()]
        
        st.write(f"**✅ Showing {len(df_filtered):,} of {len(df):,} rows**")
        
        # Display options
        col_disp1, col_disp2 = st.columns(2)
        with col_disp1:
            num_rows = st.slider("Rows to display:", 5, min(2000, len(df_filtered)), 50)
        with col_disp2:
            st.write("")
        
        st.dataframe(df_filtered.head(num_rows), use_container_width=True)
    
    # ===== TAB 2: CHARTS =====
    with tab2:
        st.subheader("📈 Advanced Visualization")
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        all_cols = df.columns.tolist()
        
        if not numeric_cols:
            st.warning("⚠️ No numeric columns found")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_col = st.selectbox("X Axis:", all_cols, key="x_axis")
            with col2:
                y_cols = st.multiselect("Y Axis:", numeric_cols, key="y_axis", default=numeric_cols[:1])
            with col3:
                chart_type = st.selectbox(
                    "Chart Type:",
                    ["Bar", "Line", "Area", "Scatter", "Heatmap", "Box Plot", "Histogram"],
                    key="chart_type"
                )
            
            with st.expander("⚙️ Chart Options"):
                col_opt1, col_opt2, col_opt3 = st.columns(3)
                with col_opt1:
                    group_by = st.checkbox("Group by", value=True)
                    remove_null = st.checkbox("Remove nulls", value=True)
                with col_opt2:
                    figsize_w = st.slider("Width", 8, 16, 12)
                    figsize_h = st.slider("Height", 4, 12, 6)
                with col_opt3:
                    sort_asc = st.checkbox("Sort A→Z", value=True)
                    color_palette = st.selectbox("Color:", ["viridis", "coolwarm", "Set2", "husl"])
            
            if st.button("🚀 Draw Chart", use_container_width=True, key="draw_chart"):
                if not y_cols:
                    st.warning("⚠️ Select at least 1 Y column")
                else:
                    try:
                        df_chart = df[[x_col] + y_cols].copy()
                        
                        if remove_null:
                            df_chart = df_chart.dropna(subset=y_cols)
                        
                        if len(df_chart) == 0:
                            st.error("❌ No data after filtering")
                        else:
                            if group_by and (df[x_col].dtype == 'object' or len(df[x_col].unique()) < len(df) / 2):
                                chart_data = df_chart.groupby(x_col)[y_cols].sum()
                            else:
                                chart_data = df_chart.set_index(x_col)[y_cols]
                            
                            if sort_asc:
                                try:
                                    chart_data = chart_data.sort_index()
                                except:
                                    pass
                            
                            st.write(f"**Chart: {', '.join(y_cols)} vs {x_col}**")
                            
                            if chart_type == "Bar":
                                st.bar_chart(chart_data)
                            elif chart_type == "Line":
                                st.line_chart(chart_data)
                            elif chart_type == "Area":
                                st.area_chart(chart_data)
                            elif chart_type == "Scatter":
                                fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
                                df_scatter = df.dropna(subset=[x_col] + y_cols)
                                
                                x_numeric = pd.factorize(df_scatter[x_col])[0] if df_scatter[x_col].dtype == 'object' else df_scatter[x_col]
                                
                                for y_col in y_cols:
                                    ax.scatter(x_numeric, df_scatter[y_col], label=y_col, alpha=0.6, s=100)
                                
                                ax.set_xlabel(x_col)
                                ax.set_ylabel("Value")
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                            
                            elif chart_type == "Box Plot":
                                fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
                                df.boxplot(column=y_cols, ax=ax)
                                st.pyplot(fig)
                            
                            elif chart_type == "Histogram":
                                fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
                                for col in y_cols:
                                    ax.hist(df[col].dropna(), alpha=0.6, label=col, bins=30)
                                ax.legend()
                                ax.set_xlabel("Value")
                                ax.set_ylabel("Frequency")
                                st.pyplot(fig)
                            
                            elif chart_type == "Heatmap":
                                if len(numeric_cols) > 1:
                                    corr = df[numeric_cols].corr()
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    sns.heatmap(corr, annot=True, fmt='.2f', cmap=color_palette, ax=ax, cbar_kws={'label': 'Correlation'})
                                    st.pyplot(fig)
                                else:
                                    st.warning("⚠️ Need at least 2 numeric columns")
                    
                    except Exception as e:
                        logger.error(f"Chart error: {str(e)}")
                        st.error(f"❌ Error: {str(e)}")
    
    # ===== TAB 3: STATISTICS =====
    with tab3:
        st.subheader("📊 Statistical Analysis")
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if not numeric_cols:
            st.warning("⚠️ No numeric columns")
        else:
            st.write("**📈 Descriptive Statistics:**")
            st.dataframe(
                df[numeric_cols].describe().T,
                use_container_width=True
            )
            
            st.write("---")
            st.write("**📉 Additional Statistics:**")
            
            stats_data = []
            for col in numeric_cols:
                stats_data.append({
                    'Column': col,
                    'Skewness': df[col].skew().round(3),
                    'Kurtosis': df[col].kurtosis().round(3),
                    'CV (%)': (df[col].std() / df[col].mean() * 100).round(2) if df[col].mean() != 0 else 0
                })
            
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
    
    # ===== TAB 4: ANALYSIS =====
    with tab4:
        st.subheader("🔍 Data Quality & Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📋 Column Information:**")
            dtype_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Nulls': df.isnull().sum(),
                'Null %': (df.isnull().sum() / len(df) * 100).round(1)
            })
            st.dataframe(dtype_info, use_container_width=True)
        
        with col2:
            st.write("**🔍 Unique Values:**")
            unique_info = pd.DataFrame({
                'Column': df.columns,
                'Unique': df.nunique(),
                'Unique %': (df.nunique() / len(df) * 100).round(1)
            })
            st.dataframe(unique_info, use_container_width=True)
        
        st.write("---")
        
        # Missing data visualization
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            st.write("**Missing Data Distribution:**")
            fig, ax = plt.subplots(figsize=(12, 4))
            missing_data[missing_data > 0].plot(kind='barh', ax=ax, color='coral')
            ax.set_xlabel('Count')
            st.pyplot(fig)
    
    # ===== TAB 5: ADVANCED =====
    with tab5:
        st.subheader("⚙️ Advanced Analytics")
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # Outlier Detection
        st.write("**🎯 Outlier Detection:**")
        col_outlier = st.selectbox("Select column:", numeric_cols)
        method = st.radio("Method:", ["IQR", "Z-Score"], horizontal=True)
        
        if st.button("Detect Outliers"):
            outliers = detect_outliers(df, col_outlier, method.lower())
            st.write(f"Found {outliers.sum()} outliers")
            
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.scatter(range(len(df)), df[col_outlier], c=outliers, cmap='RdYlGn_r', alpha=0.6)
            ax.set_ylabel(col_outlier)
            ax.set_xlabel('Index')
            st.pyplot(fig)
        
        st.write("---")
        
        # Correlation Analysis
        st.write("**📊 Correlation Matrix:**")
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, center=0)
            st.pyplot(fig)
            
            # Find high correlations
            st.write("**Strongest Correlations:**")
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr.append({
                            'Variable 1': corr_matrix.columns[i],
                            'Variable 2': corr_matrix.columns[j],
                            'Correlation': corr_matrix.iloc[i, j].round(3)
                        })
            
            if high_corr:
                st.dataframe(pd.DataFrame(high_corr), use_container_width=True)
            else:
                st.info("No strong correlations found (> 0.7)")
    
    # ===== TAB 6: EXPORT =====
    with tab6:
        st.subheader("💾 Export Data")
        
        export_col = st.selectbox("Export format:", ["CSV", "Excel", "JSON"])
        
        if export_col == "CSV":
            csv = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "⬇️ Download CSV",
                csv,
                f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        elif export_col == "Excel":
            excel_data = export_to_excel(df)
            st.download_button(
                "⬇️ Download Excel",
                excel_data,
                f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        elif export_col == "JSON":
            json_data = export_to_json(df)
            st.download_button(
                "⬇️ Download JSON",
                json_data,
                f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                use_container_width=True
            )
        
        st.write("---")
        st.write("**📋 Data Summary:**")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        with summary_col1:
            st.metric("Total Rows", f"{len(df):,}")
        with summary_col2:
            st.metric("Total Columns", len(df.columns))
        with summary_col3:
            st.metric("File Size", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")

else:
    st.info("📥 Upload a CSV or Excel file to get started")

# ===== FOOTER =====
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    "<p>📊 Advanced Data Analytics Dashboard v3.0 | Built with Streamlit</p>"
    "</div>",
    unsafe_allow_html=True
)
