"""
🎓 Educational Data Dashboard - Robust Version
Xử lý tất cả trường hợp None/lỗi
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Optional, Tuple, List
import logging
import io
import chardet

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

# ===== ROBUST DATA HANDLER =====
class RobustDataHandler:
    """Xử lý dữ liệu - Xử lý tất cả lỗi"""
    
    @staticmethod
    def detect_encoding(file) -> str:
        """Phát hiện encoding của file"""
        try:
            file.seek(0)
            raw_data = file.read(10000)
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            file.seek(0)
            return encoding if encoding else 'utf-8'
        except:
            return 'utf-8'
    
    @staticmethod
    def detect_separator(file) -> str:
        """Phát hiện separator của CSV"""
        try:
            file.seek(0)
            sample = file.read(1024).decode('utf-8', errors='ignore')
            file.seek(0)
            
            separators = [',', ';', '\t', '|']
            for sep in separators:
                if sep in sample:
                    return sep
            return ','
        except:
            return ','
    
    @staticmethod
    @st.cache_data
    def load_file(file) -> Optional[pd.DataFrame]:
        """Load file với xử lý lỗi toàn diện"""
        try:
            if not file:
                st.error("❌ Chưa chọn file")
                return None
            
            st.info("⏳ Đang tải file...")
            
            # ===== CSV =====
            if file.name.endswith('.csv'):
                try:
                    # Cách 1: Mặc định
                    file.seek(0)
                    df = pd.read_csv(file, low_memory=False)
                    
                    if df.empty or df.isnull().all().all():
                        raise ValueError("DataFrame trống")
                    
                    logger.info(f"✅ CSV loaded (default): {df.shape}")
                    st.success(f"✅ CSV loaded: {df.shape}")
                    return df
                
                except Exception as e1:
                    logger.warning(f"Default CSV load failed: {str(e1)}")
                    
                    try:
                        # Cách 2: Detect encoding
                        encoding = RobustDataHandler.detect_encoding(file)
                        logger.info(f"Trying encoding: {encoding}")
                        file.seek(0)
                        df = pd.read_csv(file, low_memory=False, encoding=encoding)
                        
                        if df.empty or df.isnull().all().all():
                            raise ValueError("DataFrame trống")
                        
                        logger.info(f"✅ CSV loaded (encoding={encoding}): {df.shape}")
                        st.success(f"✅ CSV loaded ({encoding}): {df.shape}")
                        return df
                    
                    except Exception as e2:
                        logger.warning(f"Encoding load failed: {str(e2)}")
                        
                        try:
                            # Cách 3: Detect separator
                            separator = RobustDataHandler.detect_separator(file)
                            logger.info(f"Trying separator: '{separator}'")
                            file.seek(0)
                            df = pd.read_csv(file, sep=separator, low_memory=False)
                            
                            if df.empty or df.isnull().all().all():
                                raise ValueError("DataFrame trống")
                            
                            logger.info(f"✅ CSV loaded (sep='{separator}'): {df.shape}")
                            st.success(f"✅ CSV loaded (sep='{separator}'): {df.shape}")
                            return df
                        
                        except Exception as e3:
                            logger.warning(f"Separator load failed: {str(e3)}")
                            
                            try:
                                # Cách 4: Encoding latin-1
                                logger.info("Trying encoding: latin-1")
                                file.seek(0)
                                df = pd.read_csv(file, low_memory=False, encoding='latin-1')
                                
                                if df.empty or df.isnull().all().all():
                                    raise ValueError("DataFrame trống")
                                
                                logger.info(f"✅ CSV loaded (latin-1): {df.shape}")
                                st.success(f"✅ CSV loaded (latin-1): {df.shape}")
                                return df
                            
                            except Exception as e4:
                                logger.error(f"CSV load failed all methods: {str(e4)}")
                                st.error(f"❌ Không thể load CSV: {str(e4)}")
                                return None
            
            # ===== EXCEL =====
            elif file.name.endswith(('.xlsx', '.xls')):
                try:
                    file.seek(0)
                    df = pd.read_excel(file)
                    
                    if df.empty or df.isnull().all().all():
                        raise ValueError("DataFrame trống")
                    
                    logger.info(f"✅ Excel loaded: {df.shape}")
                    st.success(f"✅ Excel loaded: {df.shape}")
                    return df
                
                except Exception as e:
                    logger.error(f"Excel load error: {str(e)}")
                    st.error(f"❌ Không thể load Excel: {str(e)}")
                    return None
            
            else:
                st.error("❌ File type không được hỗ trợ")
                return None
        
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            st.error(f"❌ Lỗi không mong muốn: {str(e)}")
            return None
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Làm sạch dữ liệu"""
        try:
            # Xóa cột trống hoàn toàn
            df = df.dropna(axis=1, how='all')
            
            # Xóa hàng trống hoàn toàn
            df = df.dropna(axis=0, how='all')
            
            # Reset index
            df = df.reset_index(drop=True)
            
            # Rename columns - loại bỏ spaces
            df.columns = df.columns.str.strip()
            
            logger.info(f"✅ Data cleaned: {df.shape}")
            return df
        
        except Exception as e:
            logger.error(f"Clean error: {str(e)}")
            return df
    
    @staticmethod
    def convert_types(df: pd.DataFrame) -> pd.DataFrame:
        """Chuyển đổi kiểu dữ liệu thông minh"""
        try:
            df_converted = df.copy()
            
            for col in df_converted.columns:
                try:
                    # Bỏ qua cột trống hoàn toàn
                    if df_converted[col].isnull().all():
                        continue
                    
                    # Convert object columns
                    if df_converted[col].dtype == 'object':
                        # Clean
                        df_converted[col] = df_converted[col].astype(str).str.strip()
                        
                        # Remove 'None' strings
                        df_converted[col] = df_converted[col].replace('None', np.nan)
                        df_converted[col] = df_converted[col].replace('none', np.nan)
                        df_converted[col] = df_converted[col].replace('', np.nan)
                        
                        # Try numeric
                        try:
                            numeric_col = pd.to_numeric(
                                df_converted[col].str.replace(',', '', regex=False),
                                errors='coerce'
                            )
                            # If most values converted, use it
                            if numeric_col.notna().sum() / len(numeric_col) > 0.5:
                                df_converted[col] = numeric_col
                                continue
                        except:
                            pass
                        
                        # Try datetime
                        try:
                            datetime_col = pd.to_datetime(
                                df_converted[col],
                                errors='coerce'
                            )
                            # If most values converted, use it
                            if datetime_col.notna().sum() / len(datetime_col) > 0.5:
                                df_converted[col] = datetime_col
                                continue
                        except:
                            pass
                
                except Exception as col_error:
                    logger.warning(f"Column {col} conversion failed: {str(col_error)}")
                    continue
            
            logger.info("✅ Types converted")
            return df_converted
        
        except Exception as e:
            logger.error(f"Convert types error: {str(e)}")
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
        numeric_count = len(df.select_dtypes(include=['float64', 'int64']).columns)
        st.metric("🔢 Số cột", numeric_count)
    with col4:
        object_count = len(df.select_dtypes(include=['object']).columns)
        st.metric("📝 Chữ cột", object_count)
    
    # Search
    col_search, val_search = st.columns([1, 2])
    with col_search:
        search_col = st.selectbox("🔍 Tìm kiếm:", df.columns, key="search_col")
    
    with val_search:
        search_val = st.text_input("Giá trị:", key="search_val")
    
    if search_val and search_col:
        try:
            df_search = df[df[search_col].astype(str).str.contains(search_val, case=False, na=False)]
            st.write(f"✅ Tìm được {len(df_search)} kết quả:")
            st.dataframe(df_search, width='stretch', height=400)
        except:
            st.warning("⚠️ Không tìm được")
    else:
        st.write("**Dữ liệu (20 dòng đầu):**")
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
    
    if not numeric_cols:
        st.warning("⚠️ Không có cột số để vẽ biểu đồ")
        return
    
    with col1:
        x_col = st.selectbox("Trục X:", all_cols, key="chart_x")
    
    with col2:
        y_cols = st.multiselect("Trục Y:", numeric_cols, key="chart_y")
    
    with col3:
        chart_type = st.selectbox(
            "Loại:", ["📊 Cột", "📈 Đường", "📉 Vùng", "🔵 Phân tán", "📐 Heatmap"],
            key="chart_type"
        )
    
    with st.expander("⚙️ Tùy chọn"):
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        
        with col_opt1:
            use_groupby = st.checkbox("Gom nhóm", value=True, key="groupby")
            remove_nulls = st.checkbox("Xóa trống", value=False, key="remove_null")
        
        with col_opt2:
            figsize_w = st.slider("Rộng", 6, 16, 10, key="width")
            figsize_h = st.slider("Cao", 4, 12, 6, key="height")
        
        with col_opt3:
            sort_asc = st.checkbox("A→Z", value=True, key="sort")
    
    if st.button("🚀 Vẽ biểu đồ", use_container_width=True):
        if not y_cols:
            st.warning("⚠️ Chọn ít nhất 1 cột Y")
            return
        
        try:
            df_chart = df[[x_col] + y_cols].copy()
            
            if remove_nulls:
                df_chart = df_chart.dropna(subset=y_cols)
            
            if len(df_chart) == 0:
                st.error("❌ Không có dữ liệu")
                return
            
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
                    ax.scatter(x_numeric, df_scatter[y_col], label=y_col, alpha=0.6, s=100)
                
                ax.set_xlabel(x_col)
                ax.set_ylabel("Giá trị")
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
            
            elif "Heatmap" in chart_type:
                if len(numeric_cols) > 1:
                    corr = df[numeric_cols].corr()
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.warning("⚠️ Cần ít nhất 2 cột số")
            
            with st.expander("📊 Dữ liệu"):
                st.dataframe(chart_data, width='stretch')
        
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
    
    st.write("**Thống kê chi tiết:**")
    stats_df = df[numeric_cols].describe().T
    st.dataframe(stats_df, width='stretch')

# ===== ANALYSIS TAB =====
def render_analysis_tab(df: pd.DataFrame):
    """Tab phân tích"""
    st.header("🔍 Phân Tích")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Kiểu dữ liệu:**")
        dtype_info = pd.DataFrame({
            'Cột': df.columns,
            'Kiểu': df.dtypes.astype(str),
            'Trống': df.isnull().sum(),
            'Trống %': (df.isnull().sum() / len(df) * 100).round(1)
        })
        st.dataframe(dtype_info, width='stretch', height=400)
    
    with col2:
        st.write("**Giá trị duy nhất:**")
        unique_info = pd.DataFrame({
            'Cột': df.columns,
            'Unique': df.nunique(),
            '% Duy nhất': (df.nunique() / len(df) * 100).round(1)
        })
        st.dataframe(unique_info, width='stretch', height=400)

# ===== MAIN =====
def main():
    """Hàm chính"""
    setup_page()
    
    st.sidebar.header("📁 Dữ liệu")
    source = st.sidebar.radio("Nguồn:", ["📤 Upload", "🔗 GitHub"], key="data_source")
    
    df = None
    
    if source == "📤 Upload":
        file = st.sidebar.file_uploader(
            "Chọn file",
            type=Config.ALLOWED_EXTENSIONS,
            key="file_uploader"
        )
        if file:
            df = RobustDataHandler.load_file(file)
    
    else:
        url = st.sidebar.text_input("URL:", key="github_url")
        if st.sidebar.button("Tải", key="load_github"):
            try:
                df = pd.read_csv(url)
                st.success(f"✅ Loaded: {df.shape}")
            except Exception as e:
                st.error(f"❌ {str(e)}")
    
    if df is not None:
        # Clean & convert
        df = RobustDataHandler.clean_data(df)
        df = RobustDataHandler.convert_types(df)
        
        if df.empty:
            st.error("❌ DataFrame rỗng sau xử lý")
            return
        
        st.success(f"✅ Ready: {len(df):,} dòng × {len(df.columns)} cột")
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Dữ liệu", "📈 Biểu đồ", "📊 Thống kê", "🔍 Phân tích"])
        
        with tab1:
            render_data_tab(df)
        
        with tab2:
            render_chart_tab(df)
        
        with tab3:
            render_stats_tab(df)
        
        with tab4:
            render_analysis_tab(df)
    
    else:
        st.info("📥 Upload file hoặc nhập URL")
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 12px;'>"
        "<p>🎓 Robust Dashboard | Xử lý tất cả None/Lỗi</p>"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
