"""
📊 Professional Data Analytics Dashboard
- Upload CSV/Excel with robust error handling
- Search & Filter Data
- Interactive Charts (Bar, Line, Area, Scatter, Heatmap)
- Statistics & Analysis
- Support 5-2000 rows display
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# ===== LOGGING =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== CONFIG =====
class Config:
    PAGE_TITLE = "Data Analytics Dashboard"
    PAGE_ICON = "📊"
    LAYOUT = "wide"
    MAX_ROWS = 2000
    ALLOWED_EXTENSIONS = ['csv', 'xlsx', 'xls']

# ===== PAGE SETUP =====
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout=Config.LAYOUT
)
st.title(f"{Config.PAGE_ICON} {Config.PAGE_TITLE}")
st.markdown("---")

# ===== LOAD FILE FUNCTION =====
@st.cache_data
def load_file(file):
    """Load CSV or Excel file with error handling"""
    try:
        st.info("⏳ Loading file...")
        
        if file.name.endswith('.csv'):
            try:
                file.seek(0)
                df = pd.read_csv(file, low_memory=False)
                logger.info(f"✅ CSV loaded: {df.shape}")
                st.success(f"✅ CSV loaded: {df.shape}")
                return df
            except Exception as e:
                try:
                    file.seek(0)
                    df = pd.read_csv(file, low_memory=False, encoding='latin-1')
                    logger.info(f"✅ CSV loaded (latin-1): {df.shape}")
                    st.success(f"✅ CSV loaded (latin-1): {df.shape}")
                    return df
                except Exception as e2:
                    logger.error(f"CSV error: {str(e2)}")
                    st.error(f"❌ Cannot load CSV: {str(e2)}")
                    return None
        else:
            file.seek(0)
            df = pd.read_excel(file)
            logger.info(f"✅ Excel loaded: {df.shape}")
            st.success(f"✅ Excel loaded: {df.shape}")
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
    """Convert data types"""
    df_converted = df.copy()
    
    for col in df_converted.columns:
        if df_converted[col].isnull().all():
            continue
        
        if df_converted[col].dtype == 'object':
            df_converted[col] = df_converted[col].astype(str).str.strip()
            df_converted[col] = df_converted[col].replace(['None', 'none', ''], np.nan)
            
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

# ===== SIDEBAR =====
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Choose file:",
    type=Config.ALLOWED_EXTENSIONS
)

df = None
if uploaded_file:
    df = load_file(uploaded_file)

# ===== MAIN CONTENT =====
if df is not None:
    df = clean_data(df)
    df = convert_types(df)
    
    st.success(f"✅ Ready: {len(df):,} rows × {len(df.columns)} columns")
    
    # ===== TABS =====
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data", "📈 Charts", "📊 Statistics", "🔍 Analysis"])
    
    # ===== TAB 1: DATA =====
    with tab1:
        st.subheader("🔍 Search & Filter")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Rows", f"{len(df):,}")
        with col2:
            st.metric("📈 Columns", len(df.columns))
        with col3:
            numeric_count = len(df.select_dtypes(include=['float64', 'int64']).columns)
            st.metric("🔢 Numeric", numeric_count)
        with col4:
            object_count = len(df.select_dtypes(include=['object']).columns)
            st.metric("📝 Text", object_count)
        
        st.write("---")
        
        # Search
        col_search, val_search = st.columns([1, 2])
        with col_search:
            search_col = st.selectbox("Column:", df.columns, key="search_col")
        with val_search:
            search_val = st.text_input("Search value:", key="search_val")
        
        # Apply filter
        if search_val:
            df_filtered = df[df[search_col].astype(str).str.contains(search_val, case=False, na=False)]
            st.write(f"✅ Found {len(df_filtered)} results")
        else:
            df_filtered = df
        
        # Rows to display
        max_rows = min(Config.MAX_ROWS, len(df_filtered))
        num_rows = st.slider("Rows to display:", 5, max_rows, 20)
        
        # Display data
        st.write(f"**Showing {num_rows} rows:**")
        st.dataframe(df_filtered.head(num_rows), use_container_width=True)
        
        # Download
        csv = df_filtered.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            "⬇️ Download CSV",
            csv,
            f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    # ===== TAB 2: CHARTS =====
    with tab2:
        st.subheader("📈 Create Charts")
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        all_cols = df.columns.tolist()
        
        if not numeric_cols:
            st.warning("⚠️ No numeric columns found")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_col = st.selectbox("X Axis:", all_cols, key="x_axis")
            
            with col2:
                y_cols = st.multiselect("Y Axis:", numeric_cols, key="y_axis")
            
            with col3:
                chart_type = st.selectbox(
                    "Chart Type:",
                    ["Bar", "Line", "Area", "Scatter", "Heatmap"],
                    key="chart_type"
                )
            
            # Advanced options
            with st.expander("⚙️ Options"):
                col_opt1, col_opt2, col_opt3 = st.columns(3)
                with col_opt1:
                    group_by = st.checkbox("Group by", value=True)
                    remove_null = st.checkbox("Remove nulls", value=False)
                with col_opt2:
                    figsize_w = st.slider("Width", 6, 16, 10)
                    figsize_h = st.slider("Height", 4, 12, 6)
                with col_opt3:
                    sort_asc = st.checkbox("Sort A→Z", value=True)
            
            # Draw chart
            if st.button("🚀 Draw Chart", use_container_width=True):
                if not y_cols:
                    st.warning("⚠️ Select at least 1 Y column")
                else:
                    try:
                        df_chart = df[[x_col] + y_cols].copy()
                        
                        if remove_null:
                            df_chart = df_chart.dropna(subset=y_cols)
                        
                        if len(df_chart) == 0:
                            st.error("❌ No data")
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
                                
                                if df_scatter[x_col].dtype == 'object':
                                    x_numeric = pd.factorize(df_scatter[x_col])[0]
                                else:
                                    x_numeric = df_scatter[x_col]
                                
                                for y_col in y_cols:
                                    ax.scatter(x_numeric, df_scatter[y_col], label=y_col, alpha=0.6, s=100)
                                
                                ax.set_xlabel(x_col)
                                ax.set_ylabel("Value")
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                                plt.close(fig)
                            
                            elif chart_type == "Heatmap":
                                if len(numeric_cols) > 1:
                                    corr = df[numeric_cols].corr()
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
                                    st.pyplot(fig)
                                    plt.close(fig)
                                else:
                                    st.warning("⚠️ Need at least 2 numeric columns")
                            
                            with st.expander("📊 View Data"):
                                st.dataframe(chart_data, use_container_width=True)
                    
                    except Exception as e:
                        logger.error(f"Chart error: {str(e)}")
                        st.error(f"❌ {str(e)}")
    
    # ===== TAB 3: STATISTICS =====
    with tab3:
        st.subheader("📊 Statistics")
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if not numeric_cols:
            st.warning("⚠️ No numeric columns")
        else:
            st.write("**Summary Statistics:**")
            st.dataframe(
                df[numeric_cols].describe().T,
                use_container_width=True
            )
    
    # ===== TAB 4: ANALYSIS =====
    with tab4:
        st.subheader("🔍 Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Column Info:**")
            dtype_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Nulls': df.isnull().sum(),
                'Null %': (df.isnull().sum() / len(df) * 100).round(1)
            })
            st.dataframe(dtype_info, use_container_width=True)
        
        with col2:
            st.write("**Unique Values:**")
            unique_info = pd.DataFrame({
                'Column': df.columns,
                'Unique': df.nunique(),
                'Unique %': (df.nunique() / len(df) * 100).round(1)
            })
            st.dataframe(unique_info, use_container_width=True)

else:
    st.info("📥 Upload a file to get started")

# ===== FOOTER =====
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    "<p>📊 Data Analytics Dashboard v2.0 | Built with Streamlit</p>"
    "</div>",
    unsafe_allow_html=True
)
