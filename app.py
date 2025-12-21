"""
📊 Professional Data Analytics Dashboard
Author: Data Engineer
Version: 2.0
Last Updated: 2025-12-21

Best Practices:
- Type hints for all functions
- Comprehensive error handling
- Logging for debugging
- Configuration management
- Clean code structure
- Input validation
- Caching optimization
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Optional, Tuple, List
import logging
from abc import ABC, abstractmethod

# ===== LOGGING CONFIGURATION =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===== CONFIGURATION =====
class Config:
    """Cấu hình ứng dụng"""
    PAGE_TITLE = "Dashboard Phân Tích Dữ Liệu"
    PAGE_ICON = "📊"
    LAYOUT = "wide"
    
    # Giới hạn dữ liệu
    MAX_ROWS = 100000
    MAX_COLS = 100
    
    # Chart settings
    DEFAULT_FIGSIZE = (10, 6)
    CHART_COLORS = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
    
    # Validation
    ALLOWED_EXTENSIONS = ['csv', 'xlsx', 'xls']
    REQUIRED_COLUMNS = []

# ===== BASE CLASS HANDLER =====
class BaseDataHandler(ABC):
    """Base class cho tất cả handlers"""
    
    @abstractmethod
    def load(self) -> Optional[pd.DataFrame]:
        """Load dữ liệu"""
        pass
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
        """Kiểm tra tính hợp lệ của DataFrame"""
        try:
            if df is None:
                return False, "DataFrame là None"
            
            if df.empty:
                return False, "DataFrame rỗng"
            
            if len(df) > Config.MAX_ROWS:
                return False, f"Vượt quá {Config.MAX_ROWS} dòng"
            
            if len(df.columns) > Config.MAX_COLS:
                return False, f"Vượt quá {Config.MAX_COLS} cột"
            
            return True, "Valid"
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False, f"Lỗi: {str(e)}"

# ===== FILE UPLOAD HANDLER =====
class FileUploadHandler(BaseDataHandler):
    """Xử lý upload file"""
    
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file
    
    def load(self) -> Optional[pd.DataFrame]:
        """Load file CSV hoặc Excel"""
        try:
            if not self.uploaded_file:
                return None
            
            # Kiểm tra extension
            file_ext = self.uploaded_file.name.split('.')[-1].lower()
            if file_ext not in Config.ALLOWED_EXTENSIONS:
                raise ValueError(f"File type '{file_ext}' không được hỗ trợ")
            
            # Load file
            logger.info(f"Loading file: {self.uploaded_file.name}")
            
            if file_ext == 'csv':
                df = pd.read_csv(self.uploaded_file, low_memory=False)
            else:
                df = pd.read_excel(self.uploaded_file)
            
            is_valid, msg = self.validate_dataframe(df)
            if not is_valid:
                raise ValueError(msg)
            
            logger.info(f"File loaded successfully: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"File upload error: {str(e)}")
            st.error(f"❌ Lỗi đọc file: {str(e)}")
            return None

# ===== GITHUB HANDLER =====
class GitHubHandler(BaseDataHandler):
    """Xử lý tải từ GitHub"""
    
    def __init__(self, url: str):
        self.url = url
    
    def load(self) -> Optional[pd.DataFrame]:
        """Load CSV từ GitHub"""
        try:
            if not self.url or not self.url.startswith('http'):
                raise ValueError("URL không hợp lệ")
            
            logger.info(f"Loading from GitHub: {self.url}")
            df = pd.read_csv(self.url)
            
            is_valid, msg = self.validate_dataframe(df)
            if not is_valid:
                raise ValueError(msg)
            
            logger.info(f"GitHub data loaded: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"GitHub load error: {str(e)}")
            st.error(f"❌ Lỗi tải từ GitHub: {str(e)}")
            return None

# ===== SAMPLE DATA HANDLER =====
class SampleDataHandler(BaseDataHandler):
    """Tạo dữ liệu mẫu"""
    
    def load(self) -> Optional[pd.DataFrame]:
        """Tạo dữ liệu mẫu"""
        try:
            logger.info("Generating sample data")
            np.random.seed(42)
            
            data = {
                'Ngày': pd.date_range('2023-01-01', periods=100),
                'Sản phẩm': np.random.choice(
                    ['Laptop', 'Điện thoại', 'Tablet', 'Tai nghe'], 100
                ),
                'Khu vực': np.random.choice(
                    ['Hà Nội', 'TP.HCM', 'Đà Nẵng', 'Cần Thơ'], 100
                ),
                'Số lượng': np.random.randint(1, 100, 100),
                'Đơn giá': np.random.randint(500000, 5000000, 100),
            }
            
            df = pd.DataFrame(data)
            df['Doanh thu'] = df['Số lượng'] * df['Đơn giá']
            
            is_valid, msg = self.validate_dataframe(df)
            if not is_valid:
                raise ValueError(msg)
            
            logger.info(f"Sample data created: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Sample data error: {str(e)}")
            st.error(f"❌ Lỗi tạo dữ liệu mẫu: {str(e)}")
            return None

# ===== DATA PROCESSOR =====
class DataProcessor:
    """Xử lý và chuẩn bị dữ liệu"""
    
    @staticmethod
    def convert_types(df: pd.DataFrame) -> pd.DataFrame:
        """Chuyển đổi kiểu dữ liệu tự động"""
        try:
            df_converted = df.copy()
            
            for col in df_converted.columns:
                if df_converted[col].dtype == 'object':
                    df_converted[col] = df_converted[col].astype(str).str.strip()
                    
                    # Thử chuyển sang số
                    try:
                        df_converted[col] = pd.to_numeric(
                            df_converted[col].str.replace(',', '', regex=False),
                            errors='coerce'
                        )
                        continue
                    except:
                        pass
                    
                    # Thử chuyển sang datetime
                    try:
                        df_converted[col] = pd.to_datetime(
                            df_converted[col],
                            errors='coerce'
                        )
                    except:
                        pass
            
            logger.info("Data types converted successfully")
            return df_converted
            
        except Exception as e:
            logger.error(f"Type conversion error: {str(e)}")
            return df
    
    @staticmethod
    def clean_data(df: pd.DataFrame, remove_nulls: bool = True) -> Tuple[pd.DataFrame, dict]:
        """Làm sạch dữ liệu"""
        try:
            stats = {
                'initial_rows': len(df),
                'initial_cols': len(df.columns),
            }
            
            # Xóa cột trống
            df_cleaned = df.dropna(axis=1, how='all')
            stats['cols_removed'] = stats['initial_cols'] - len(df_cleaned.columns)
            
            # Xóa hàng trống
            if remove_nulls:
                df_cleaned = df_cleaned.dropna(how='all')
            
            stats['final_rows'] = len(df_cleaned)
            stats['rows_removed'] = stats['initial_rows'] - stats['final_rows']
            
            logger.info(f"Data cleaned: {stats}")
            return df_cleaned, stats
            
        except Exception as e:
            logger.error(f"Data cleaning error: {str(e)}")
            return df, {'error': str(e)}

# ===== CHART GENERATOR =====
class ChartGenerator:
    """Tạo biểu đồ"""
    
    @staticmethod
    def validate_chart_data(
        df: pd.DataFrame,
        x_col: str,
        y_cols: List[str]
    ) -> Tuple[bool, str]:
        """Kiểm tra dữ liệu biểu đồ"""
        try:
            if x_col not in df.columns:
                return False, f"Cột '{x_col}' không tồn tại"
            
            for y_col in y_cols:
                if y_col not in df.columns:
                    return False, f"Cột '{y_col}' không tồn tại"
            
            return True, "Valid"
            
        except Exception as e:
            logger.error(f"Chart validation error: {str(e)}")
            return False, str(e)
    
    @staticmethod
    def prepare_chart_data(
        df: pd.DataFrame,
        x_col: str,
        y_cols: List[str],
        use_groupby: bool = True,
        remove_nulls: bool = True
    ) -> Optional[pd.DataFrame]:
        """Chuẩn bị dữ liệu cho biểu đồ"""
        try:
            df_chart = df[[x_col] + y_cols].copy()
            
            if remove_nulls:
                df_chart = df_chart.dropna(subset=y_cols)
            
            if len(df_chart) == 0:
                raise ValueError("Không còn dữ liệu sau khi xóa giá trị trống")
            
            # Xử lý groupby
            if use_groupby and (df[x_col].dtype == 'object' or 
                               len(df[x_col].unique()) < len(df) / 2):
                chart_data = df_chart.groupby(x_col)[y_cols].sum()
            else:
                df_chart = df_chart.sort_values(x_col)
                chart_data = df_chart.set_index(x_col)[y_cols]
            
            logger.info(f"Chart data prepared: {chart_data.shape}")
            return chart_data
            
        except Exception as e:
            logger.error(f"Chart data preparation error: {str(e)}")
            st.error(f"❌ Lỗi chuẩn bị dữ liệu: {str(e)}")
            return None
    
    @staticmethod
    def plot_bar_chart(chart_data: pd.DataFrame) -> None:
        """Vẽ biểu đồ cột"""
        try:
            st.bar_chart(chart_data)
            logger.info("Bar chart created successfully")
        except Exception as e:
            logger.error(f"Bar chart error: {str(e)}")
            st.error(f"❌ Lỗi vẽ biểu đồ cột: {str(e)}")
    
    @staticmethod
    def plot_line_chart(chart_data: pd.DataFrame) -> None:
        """Vẽ biểu đồ đường"""
        try:
            st.line_chart(chart_data)
            logger.info("Line chart created successfully")
        except Exception as e:
            logger.error(f"Line chart error: {str(e)}")
            st.error(f"❌ Lỗi vẽ biểu đồ đường: {str(e)}")
    
    @staticmethod
    def plot_area_chart(chart_data: pd.DataFrame) -> None:
        """Vẽ biểu đồ vùng"""
        try:
            st.area_chart(chart_data)
            logger.info("Area chart created successfully")
        except Exception as e:
            logger.error(f"Area chart error: {str(e)}")
            st.error(f"❌ Lỗi vẽ biểu đồ vùng: {str(e)}")
    
    @staticmethod
    def plot_scatter_chart(
        df: pd.DataFrame,
        x_col: str,
        y_cols: List[str],
        figsize: Tuple[int, int]
    ) -> None:
        """Vẽ biểu đồ phân tán"""
        try:
            df_scatter = df.dropna(subset=[x_col] + y_cols)
            
            if len(df_scatter) == 0:
                st.error("❌ Không có dữ liệu hợp lệ")
                return
            
            fig, ax = plt.subplots(figsize=figsize)
            
            if df_scatter[x_col].dtype == 'object':
                x_numeric = pd.factorize(df_scatter[x_col])[0]
                x_label = x_col
            else:
                x_numeric = df_scatter[x_col]
                x_label = x_col
            
            for y_col in y_cols:
                ax.scatter(x_numeric, df_scatter[y_col], 
                          label=y_col, alpha=0.6, s=100)
            
            ax.set_xlabel(x_label)
            ax.set_ylabel("Giá trị")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            logger.info("Scatter chart created successfully")
            
        except Exception as e:
            logger.error(f"Scatter chart error: {str(e)}")
            st.error(f"❌ Lỗi vẽ biểu đồ phân tán: {str(e)}")

# ===== UI COMPONENTS =====
class UIManager:
    """Quản lý giao diện"""
    
    @staticmethod
    def setup_page() -> None:
        """Cấu hình trang"""
        st.set_page_config(
            page_title=Config.PAGE_TITLE,
            page_icon=Config.PAGE_ICON,
            layout=Config.LAYOUT
        )
        st.title(f"{Config.PAGE_ICON} {Config.PAGE_TITLE}")
        st.markdown("---")
    
    @staticmethod
    def render_data_tab(df: pd.DataFrame) -> None:
        """Render tab dữ liệu"""
        st.header("📋 Dữ liệu Chi Tiết")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Dữ liệu mẫu:**")
            st.dataframe(df.head(20), width='stretch')
        
        with col2:
            st.write("**Thông tin:**")
            st.metric("Dòng", len(df))
            st.metric("Cột", len(df.columns))
        
        # Download button
        csv = df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            "⬇️ Tải CSV",
            data=csv,
            file_name=f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    @staticmethod
    def render_chart_tab(df: pd.DataFrame) -> None:
        """Render tab biểu đồ"""
        st.header("📈 Biểu Đồ")
        
        col1, col2, col3 = st.columns(3)
        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        with col1:
            x_column = st.selectbox("Trục X:", all_cols)
        
        with col2:
            y_columns = st.multiselect("Trục Y:", numeric_cols)
        
        with col3:
            chart_type = st.selectbox(
                "Loại biểu đồ:",
                ["📊 Cột", "📈 Đường", "📉 Vùng", "🔵 Phân tán"]
            )
        
        # Tùy chọn
        with st.expander("⚙️ Tùy chọn"):
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                use_groupby = st.checkbox("Gom nhóm", value=True)
                remove_nulls = st.checkbox("Xóa trống", value=True)
            with col_opt2:
                figsize_w = st.slider("Rộng", 8, 16, 10)
                figsize_h = st.slider("Cao", 4, 12, 6)
        
        # Vẽ biểu đồ
        if st.button("🚀 Vẽ biểu đồ", use_container_width=True):
            if not y_columns:
                st.warning("⚠️ Chọn ít nhất 1 cột cho Y")
                return
            
            # Validate
            is_valid, msg = ChartGenerator.validate_chart_data(df, x_column, y_columns)
            if not is_valid:
                st.error(f"❌ {msg}")
                return
            
            # Prepare data
            chart_data = ChartGenerator.prepare_chart_data(
                df, x_column, y_columns, use_groupby, remove_nulls
            )
            
            if chart_data is None:
                return
            
            # Plot
            st.subheader(f"📊 {', '.join(y_columns)} theo {x_column}")
            
            if "Cột" in chart_type:
                ChartGenerator.plot_bar_chart(chart_data)
            elif "Đường" in chart_type:
                ChartGenerator.plot_line_chart(chart_data)
            elif "Vùng" in chart_type:
                ChartGenerator.plot_area_chart(chart_data)
            elif "Phân tán" in chart_type:
                ChartGenerator.plot_scatter_chart(df, x_column, y_columns, (figsize_w, figsize_h))
            
            # Show data
            with st.expander("📊 Dữ liệu biểu đồ"):
                st.dataframe(chart_data, width='stretch')
    
    @staticmethod
    def render_stats_tab(df: pd.DataFrame) -> None:
        """Render tab thống kê"""
        st.header("📊 Thống Kê")
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Dòng", len(df))
        with col2:
            st.metric("Cột số", len(numeric_cols))
        with col3:
            st.metric("Cột chữ", len(df.columns) - len(numeric_cols))
        with col4:
            st.metric("Tổng cột", len(df.columns))
        
        st.write("**Thống kê chi tiết:**")
        st.dataframe(df.describe().T, width='stretch')

# ===== MAIN APPLICATION =====
def main():
    """Hàm chính"""
    try:
        # Setup
        UIManager.setup_page()
        
        # Sidebar
        st.sidebar.header("📁 Dữ liệu")
        data_source = st.sidebar.radio(
            "Nguồn:",
            ["📤 Upload", "🔗 GitHub", "📋 Mẫu"]
        )
        
        # Load data
        df = None
        
        if data_source == "📤 Upload":
            file = st.sidebar.file_uploader("Chọn file", type=Config.ALLOWED_EXTENSIONS)
            if file:
                handler = FileUploadHandler(file)
                df = handler.load()
        
        elif data_source == "🔗 GitHub":
            url = st.sidebar.text_input("URL:", "https://raw.githubusercontent.com/...")
            if st.sidebar.button("Tải"):
                handler = GitHubHandler(url)
                df = handler.load()
        
        elif data_source == "📋 Mẫu":
            if st.sidebar.button("Tạo dữ liệu mẫu"):
                handler = SampleDataHandler()
                df = handler.load()
        
        # Process data
        if df is not None:
            st.success("✅ Tải dữ liệu thành công")
            
            # Clean data
            df, clean_stats = DataProcessor.clean_data(df)
            df = DataProcessor.convert_types(df)
            
            st.info(f"📊 {len(df)} dòng × {len(df.columns)} cột")
            
            # Tabs
            tab1, tab2, tab3 = st.tabs(["📋 Dữ liệu", "📈 Biểu đồ", "📊 Thống kê"])
            
            with tab1:
                UIManager.render_data_tab(df)
            
            with tab2:
                UIManager.render_chart_tab(df)
            
            with tab3:
                UIManager.render_stats_tab(df)
        
        else:
            st.info("📥 Chọn dữ liệu ở sidebar")
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: gray; font-size: 12px;'>"
            "<p>📊 Professional Dashboard v2.0 | Best Practices Applied</p>"
            "</div>",
            unsafe_allow_html=True
        )
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"❌ Lỗi ứng dụng: {str(e)}")

if __name__ == "__main__":
    main()
