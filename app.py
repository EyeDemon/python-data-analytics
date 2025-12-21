"""
📊 Educational Data Dashboard
Tối ưu cho dữ liệu học viên/sinh viên
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Optional, Tuple, List
import logging

# ===== LOGGING =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== CONFIG =====
class Config:
    PAGE_TITLE = "Dashboard Dữ Liệu Học Viên"
    PAGE_ICON = "🎓"
    LAYOUT = "wide"
    MAX_ROWS = 1000000
    ALLOWED_EXTENSIONS = ['csv', 'xlsx', 'xls']

# ===== DATA HANDLER =====
class DataHandler:
    """Xử lý dữ liệu"""
    
    @staticmethod
    @st.cache_data
    def load_file(file) -> Optional[pd.DataFrame]:
        """Load file với xử lý lỗi chi tiết"""
        try:
            if file.name.endswith('.csv'):
                # Load CSV với nhiều cách khác nhau
                try:
                    df = pd.read_csv(file, low_memory=False)
                except:
                    # Thử load lại với encoding khác
                    file.seek(0)
                    df = pd.read_csv(file, low_memory=False, encoding='latin-1')
            else:
                df = pd.read_excel(file)
            
            # Debug info
            st.write(f"**📊 Thông tin file:**")
            st.write(f"- Shape: {df.shape}")
            st.write(f"- Dtypes:\n{df.dtypes}")
            st.write(f"- Columns: {list(df.columns)}")
            st.write(f"- Nulls:\n{df.isnull().sum()}")
            
            logger.info(f"✅ Loaded: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"❌ Load error: {str(e)}")
            st.error(f"❌ Lỗi load file: {str(e)}")
            return None
    
    @staticmethod
    @st.cache_data
    def load_url(url: str) -> Optional[pd.DataFrame]:
        """Load từ URL"""
        try:
            df = pd.read_csv(url)
            logger.info(f"✅ Loaded from URL: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"❌ URL load error: {str(e)}")
            st.error(f"❌ Lỗi: {str(e)}")
            return None
    
    @staticmethod
    def convert_types(df: pd.DataFrame) -> pd.DataFrame:
        """Chuyển đổi kiểu - xử lý tốt hơn"""
        try:
            df_converted = df.copy()
            
            for col in df_converted.columns:
                # Bỏ qua cột trống hoàn toàn
                if df_converted[col].isnull().all():
                    continue
                
                if df_converted[col].dtype == 'object':
                    # Clean whitespace
                    df_converted[col] = df_converted[col].astype(str).str.strip()
                    
                    # Skip if all 'None' string
                    if (df_converted[col] == 'None').all():
                        continue
                    
                    # Try numeric
                    try:
                        df_converted[col] = pd.to_numeric(
                            df_converted[col].str.replace(',', '', regex=False),
                            errors='coerce'
                        )
                        continue
                    except:
                        pass
                    
                    # Try datetime
                    try:
                        df_converted[col] = pd.to_datetime(
                            df_converted[col],
                            errors='coerce'
                        )
                    except:
                        pass
            
            logger.info("✅ Types converted")
            return df_converted
        except Exception as e:
            logger.error(f"❌ Conversion error: {str(e)}")
            return df

# ===== PAGE SETUP =====
def setup_page():
    """Setup trang"""
    st.set_page_config(
        page_title=Config.PAGE_TITLE,
        page_icon=Config.PAGE_ICON,
        layout=Config.LAYOUT
    )
    st.title(f"{Config.PAGE_ICON} {Config.PAGE_TITLE}")
    st.markdown("---")

# ===== DATA TAB =====
def render_data_tab(df: pd.DataFrame):
    """Tab dữ liệu"""
    st.header("📋 Dữ Liệu Chi Tiết")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Dòng", f"{len(df):,}")
    with col2:
        st.metric("📈 Cột", len(df.columns))
    with col3:
        st.metric("🔢 Số cột", len(df.select_dtypes(include=['float64', 'int64']).columns))
    with col4:
        st.metric("📝 Cột chữ", len(df.select_dtypes(include=['object']).columns))
    
    # Search
    search_col = st.selectbox("🔍 Tìm kiếm theo cột:", df.columns)
    search_val = st.text_input("Nhập giá trị:")
    
    if search_val:
        df_search = df[df[search_col].astype(str).str.contains(search_val, case=False, na=False)]
        st.write(f"Tìm được {len(df_search)} kết quả:")
        st.dataframe(df_search, width='stretch', height=400)
    else:
        st.write("**Dữ liệu mẫu (20 dòng đầu):**")
        st.dataframe(df.head(20), width='stretch', height=400)
    
    # Download
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        "⬇️ Tải CSV",
        data=csv,
        file_name=f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# ===== CHART TAB =====
def render_chart_tab(df: pd.DataFrame):
    """Tab biểu đồ"""
    st.header("📈 Biểu Đồ")
    
    col1, col2, col3 = st.columns(3)
    
    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    with col1:
        x_col = st.selectbox("Trục X:", all_cols)
    
    with col2:
        y_cols = st.multiselect("Trục Y (số):", numeric_cols)
    
    with col3:
        chart_type = st.selectbox(
            "Loại:",
            ["📊 Cột", "📈 Đường", "📉 Vùng", "🔵 Phân tán", "📐 Heatmap"]
        )
    
    # Options
    with st.expander("⚙️ Tùy chọn"):
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        
        with col_opt1:
            use_groupby = st.checkbox("Gom nhóm", value=True)
            remove_nulls = st.checkbox("Xóa trống", value=False)
        
        with col_opt2:
            figsize_w = st.slider("Rộng", 6, 16, 10)
            figsize_h = st.slider("Cao", 4, 12, 6)
        
        with col_opt3:
            sort_asc = st.checkbox("Sắp xếp A→Z", value=True)
            show_values = st.checkbox("Hiện giá trị", value=False)
    
    # Draw
    if st.button("🚀 Vẽ biểu đồ", use_container_width=True):
        if not y_cols:
            st.warning("⚠️ Chọn ít nhất 1 cột Y")
            return
        
        try:
            # Prepare
            df_chart = df[[x_col] + y_cols].copy()
            
            if remove_nulls:
                df_chart = df_chart.dropna(subset=y_cols)
            
            if len(df_chart) == 0:
                st.error("❌ Không có dữ liệu")
                return
            
            # Process
            if use_groupby and (df[x_col].dtype == 'object' or 
                               len(df[x_col].unique()) < len(df) / 2):
                chart_data = df_chart.groupby(x_col)[y_cols].sum()
            else:
                df_chart = df_chart.sort_values(x_col)
                chart_data = df_chart.set_index(x_col)[y_cols]
            
            if sort_asc:
                try:
                    chart_data = chart_data.sort_index()
                except:
                    pass
            
            # Plot
            st.subheader(f"📊 {', '.join(y_cols)} theo {x_col}")
            
            if "Cột" in chart_type:
                st.bar_chart(chart_data)
            
            elif "Đường" in chart_type:
                st.line_chart(chart_data)
            
            elif "Vùng" in chart_type:
                st.area_chart(chart_data)
            
            elif "Phân tán" in chart_type:
                fig, ax = plt.subplots(figsize=(figsize_w, figsize_h))
                
                df_scatter = df.dropna(subset=[x_col] + y_cols)
                
                if df_scatter[x_col].dtype == 'object':
                    x_numeric = pd.factorize(df_scatter[x_col])[0]
                else:
                    x_numeric = df_scatter[x_col]
                
                for y_col in y_cols:
                    ax.scatter(x_numeric, df_scatter[y_col], 
                              label=y_col, alpha=0.6, s=100)
                
                ax.set_xlabel(x_col)
                ax.set_ylabel("Giá trị")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            elif "Heatmap" in chart_type:
                if len(numeric_cols) > 1:
                    corr = df[numeric_cols].corr()
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr, annot=True, fmt='.2f', 
                               cmap='coolwarm', ax=ax)
                    st.pyplot(fig)
                else:
                    st.warning("⚠️ Cần ít nhất 2 cột số")
            
            # Show data
            with st.expander("📊 Dữ liệu biểu đồ"):
                st.dataframe(chart_data, width='stretch')
                
                csv_chart = chart_data.to_csv(encoding='utf-8-sig')
                st.download_button(
                    "⬇️ Tải dữ liệu",
                    data=csv_chart,
                    file_name=f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            logger.error(f"Chart error: {str(e)}")
            st.error(f"❌ {str(e)}")

# ===== STATS TAB =====
def render_stats_tab(df: pd.DataFrame):
    """Tab thống kê"""
    st.header("📊 Thống Kê")
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if not numeric_cols:
        st.warning("⚠️ Không có cột số")
        return
    
    # Basic stats
    st.write("**Thống kê chi tiết:**")
    stats_df = df[numeric_cols].describe().T
    st.dataframe(stats_df, width='stretch')
    
    # Column stats
    st.write("**Thống kê từng cột:**")
    col_select = st.selectbox("Chọn cột:", numeric_cols)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Min", f"{df[col_select].min():,.0f}")
    with col2:
        st.metric("Max", f"{df[col_select].max():,.0f}")
    with col3:
        st.metric("Avg", f"{df[col_select].mean():,.0f}")
    with col4:
        st.metric("Std", f"{df[col_select].std():,.0f}")
    
    # Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df[col_select].dropna(), bins=30, color='skyblue', edgecolor='black')
    ax.set_title(f"Phân bố {col_select}")
    ax.set_xlabel("Giá trị")
    ax.set_ylabel("Tần số")
    st.pyplot(fig)

# ===== ANALYSIS TAB =====
def render_analysis_tab(df: pd.DataFrame):
    """Tab phân tích"""
    st.header("🔍 Phân Tích")
    
    col1, col2 = st.columns(2)
    
    # Categorical
    with col1:
        st.write("**Phân tích danh mục:**")
        
        # Lấy các cột danh mục
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if not cat_cols or len(cat_cols) == 0:
            st.warning("⚠️ Không có cột danh mục")
        else:
            try:
                # Selectbox với default value
                cat_col = st.selectbox(
                    "Chọn cột:",
                    cat_cols,
                    index=0,  # Chọn cột đầu tiên mặc định
                    key="analysis_cat_col"
                )
                
                # Kiểm tra cat_col hợp lệ
                if cat_col and cat_col in df.columns:
                    top_n = st.slider("Top", 5, 20, 10, key="analysis_top_n")
                    
                    # Vẽ biểu đồ
                    fig, ax = plt.subplots(figsize=(10, 6))
                    value_counts = df[cat_col].value_counts().head(top_n)
                    value_counts.plot(kind='barh', ax=ax, color='coral')
                    ax.set_title(f"Top {top_n} {cat_col}")
                    ax.set_xlabel("Số lượng")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.error("❌ Cột không hợp lệ")
            except Exception as e:
                logger.error(f"Analysis error: {str(e)}")
                st.error(f"❌ Lỗi: {str(e)}")
    
    # Data types
    with col2:
        st.write("**Kiểu dữ liệu:**")
        try:
            dtype_info = pd.DataFrame({
                'Cột': df.columns,
                'Kiểu': df.dtypes.astype(str),
                'Trống': df.isnull().sum(),
                'Trống %': (df.isnull().sum() / len(df) * 100).round(1)
            })
            st.dataframe(dtype_info, width='stretch', height=400)
        except Exception as e:
            logger.error(f"Dtype error: {str(e)}")
            st.error(f"❌ Lỗi: {str(e)}")

# ===== MAIN =====
def main():
    """Hàm chính"""
    setup_page()
    
    # Sidebar
    st.sidebar.header("📁 Dữ liệu")
    source = st.sidebar.radio("Nguồn:", ["📤 Upload", "🔗 GitHub"])
    
    df = None
    
    if source == "📤 Upload":
        file = st.sidebar.file_uploader(
            "Chọn file",
            type=Config.ALLOWED_EXTENSIONS
        )
        if file:
            df = DataHandler.load_file(file)
    
    else:
        url = st.sidebar.text_input(
            "URL:",
            "https://raw.githubusercontent.com/.../data.csv"
        )
        if st.sidebar.button("Tải"):
            df = DataHandler.load_url(url)
    
    # Process
    if df is not None:
        st.success("✅ Tải thành công")
        
        # Convert types
        df = DataHandler.convert_types(df)
        
        # Remove completely empty columns
        df = df.dropna(axis=1, how='all')
        
        st.info(f"📊 {len(df):,} dòng × {len(df.columns)} cột")
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Dữ liệu",
            "📈 Biểu đồ",
            "📊 Thống kê",
            "🔍 Phân tích"
        ])
        
        with tab1:
            render_data_tab(df)
        
        with tab2:
            render_chart_tab(df)
        
        with tab3:
            render_stats_tab(df)
        
        with tab4:
            render_analysis_tab(df)
    
    else:
        st.info("📥 Upload file hoặc nhập URL ở sidebar")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 12px;'>"
        "<p>🎓 Educational Dashboard | Optimized for Student Data</p>"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
