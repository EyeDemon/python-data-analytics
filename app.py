import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# ===== CẤU HÌNH TRANG =====
st.set_page_config(
    page_title="Dashboard Phân Tích Dữ Liệu",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .metric-card { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ===== TIÊU ĐỀ =====
st.title("📊 Dashboard Phân Tích Dữ Liệu")
st.markdown("---")

# ===== CÁC HÀM HỖ TRỢ =====
@st.cache_data
def load_data(uploaded_file):
    """Đọc file CSV hoặc Excel"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, low_memory=False)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            return None
        return df
    except Exception as e:
        st.error(f"❌ Lỗi khi đọc file: {str(e)}")
        return None

@st.cache_data
def load_csv_from_url(url):
    """Đọc CSV từ GitHub"""
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"❌ Lỗi khi tải từ URL: {str(e)}")
        return None

def auto_convert_dtypes(df):
    """Chuyển đổi kiểu dữ liệu tự động"""
    df_converted = df.copy()
    
    for col in df_converted.columns:
        if df_converted[col].dtype == 'object':
            df_converted[col] = df_converted[col].astype(str).str.strip()
            
            try:
                df_converted[col] = pd.to_numeric(
                    df_converted[col].str.replace(',', '', regex=False),
                    errors='coerce'
                )
            except:
                pass
            
            if df_converted[col].dtype == 'object':
                try:
                    df_converted[col] = pd.to_datetime(
                        df_converted[col],
                        errors='coerce'
                    )
                except:
                    pass
    
    return df_converted

# ===== SIDEBAR - NHẬP DỮ LIỆU =====
st.sidebar.header("📁 Dữ liệu đầu vào")

data_source = st.sidebar.radio(
    "Chọn nguồn dữ liệu:",
    ["📤 Upload file", "🔗 Từ GitHub", "📋 Dữ liệu mẫu"]
)

df = None

if data_source == "📤 Upload file":
    uploaded_file = st.sidebar.file_uploader(
        "Chọn file CSV/Excel",
        type=['csv', 'xlsx', 'xls']
    )
    if uploaded_file is not None:
        df = load_data(uploaded_file)

elif data_source == "🔗 Từ GitHub":
    github_url = st.sidebar.text_input(
        "Nhập URL raw từ GitHub:",
        "https://raw.githubusercontent.com/YOUR_USERNAME/REPO/main/data.csv"
    )
    if st.sidebar.button("📥 Tải dữ liệu", use_container_width=True):
        df = load_csv_from_url(github_url)

elif data_source == "📋 Dữ liệu mẫu":
    if st.sidebar.button("📥 Tải dữ liệu mẫu", use_container_width=True):
        # Tạo dữ liệu mẫu
        np.random.seed(42)
        sample_data = {
            'Ngày': pd.date_range('2023-01-01', periods=50),
            'SanPham': np.random.choice(['Laptop', 'Chuột', 'Bàn phím', 'Tai nghe'], 50),
            'SoLuong': np.random.randint(1, 100, 50),
            'DoanThu': np.random.randint(500000, 5000000, 50),
            'KhuVuc': np.random.choice(['Hà Nội', 'TP.HCM', 'Đà Nẵng', 'Cần Thơ'], 50)
        }
        df = pd.DataFrame(sample_data)

# ===== XỬ LÝ DỮ LIỆU =====
if df is not None:
    st.success(f"✅ Đã tải dữ liệu thành công!")
    st.info(f"📊 Kích thước: {df.shape[0]} dòng × {df.shape[1]} cột")
    
    # Chuyển đổi kiểu dữ liệu
    df = auto_convert_dtypes(df)
    df = df.dropna(how='all')
    
    # ===== TAB 1: DỮ LIỆU =====
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Dữ liệu",
        "📈 Biểu đồ",
        "📊 Thống kê",
        "🔍 Phân tích"
    ])
    
    # --- TAB 1: DỮ LIỆU ---
    with tab1:
        st.header("Dữ liệu chi tiết")
        
        col_preview, col_info = st.columns([2, 1])
        
        with col_preview:
            st.write("**Dữ liệu mẫu (20 dòng đầu):**")
            st.dataframe(df.head(20), width='stretch')
        
        with col_info:
            st.write("**Thông tin cơ bản:**")
            st.metric("Tổng dòng", len(df))
            st.metric("Tổng cột", df.shape[1])
            st.metric("Kiểu dữ liệu", len(df.dtypes))
        
        # Tải file
        csv = df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="⬇️ Tải CSV",
            data=csv,
            file_name=f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # --- TAB 2: BIỂU ĐỒ ---
    with tab2:
        st.header("Tùy chỉnh biểu đồ")
        
        col1, col2, col3 = st.columns(3)
        all_columns = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        with col1:
            x_column = st.selectbox(
                "Trục X (Hoành):",
                all_columns,
                help="Trục ngang - danh mục hoặc thời gian"
            )
        
        with col2:
            y_columns = st.multiselect(
                "Trục Y (Tung):",
                numeric_cols,
                help="Chỉ hiện các cột số"
            )
        
        with col3:
            chart_type = st.selectbox(
                "Loại biểu đồ:",
                ["📊 Cột (Bar)", "📈 Đường (Line)", "📉 Vùng (Area)", "🔵 Phân tán (Scatter)"]
            )
        
        # Tùy chọn nâng cao
        with st.expander("⚙️ Tùy chọn nâng cao"):
            col_adv1, col_adv2, col_adv3 = st.columns(3)
            
            with col_adv1:
                use_groupby = st.checkbox("Gom nhóm dữ liệu", value=True)
                sort_ascending = st.checkbox("Sắp xếp tăng dần", value=True)
            
            with col_adv2:
                figsize_width = st.slider("Chiều rộng", 8, 16, 10)
                figsize_height = st.slider("Chiều cao", 4, 12, 6)
            
            with col_adv3:
                remove_nulls = st.checkbox("Xóa giá trị trống", value=True)
        
        if st.button("🚀 Vẽ biểu đồ", use_container_width=True):
            if len(y_columns) == 0:
                st.warning("⚠️ Chọn ít nhất 1 cột cho Trục Y")
            else:
                try:
                    # Bước 1: Kiểm tra dữ liệu đầu vào
                    st.write("**🔍 Kiểm tra dữ liệu...**")
                    
                    if x_column not in df.columns:
                        st.error(f"❌ Cột '{x_column}' không tồn tại!")
                        st.stop()
                    
                    for col in y_columns:
                        if col not in df.columns:
                            st.error(f"❌ Cột '{col}' không tồn tại!")
                            st.stop()
                    
                    # Bước 2: Tạo bản sao dữ liệu
                    try:
                        df_chart = df[[x_column] + y_columns].copy()
                        st.write(f"✅ Lấy {len(df_chart)} dòng dữ liệu")
                    except Exception as e:
                        st.error(f"❌ Lỗi khi lấy dữ liệu: {str(e)}")
                        st.stop()
                    
                    # Bước 3: Xóa giá trị NaN
                    try:
                        if remove_nulls:
                            rows_before = len(df_chart)
                            df_chart = df_chart.dropna(subset=y_columns)
                            rows_after = len(df_chart)
                            st.write(f"✅ Xóa {rows_before - rows_after} dòng trống")
                        
                        if len(df_chart) == 0:
                            st.error("❌ Không còn dữ liệu sau khi xóa giá trị trống!")
                            st.stop()
                    except Exception as e:
                        st.error(f"❌ Lỗi khi xóa giá trị trống: {str(e)}")
                        st.stop()
                    
                    # Bước 4: Xử lý dữ liệu
                    try:
                        if use_groupby and df[x_column].dtype == 'object':
                            st.write("✅ Gom nhóm theo danh mục...")
                            chart_data = df_chart.groupby(x_column)[y_columns].sum()
                        elif use_groupby and len(df[x_column].unique()) < len(df) / 2:
                            st.write("✅ Gom nhóm dữ liệu...")
                            chart_data = df_chart.groupby(x_column)[y_columns].sum()
                        else:
                            st.write("✅ Sắp xếp dữ liệu...")
                            df_chart = df_chart.sort_values(x_column)
                            chart_data = df_chart.set_index(x_column)[y_columns]
                    except Exception as e:
                        st.error(f"❌ Lỗi khi xử lý dữ liệu: {str(e)}")
                        st.write(f"**Chi tiết:** {type(e).__name__}")
                        st.stop()
                    
                    # Bước 5: Sắp xếp
                    try:
                        if sort_ascending:
                            chart_data = chart_data.sort_index()
                            st.write("✅ Sắp xếp tăng dần")
                    except Exception as e:
                        st.warning(f"⚠️ Không thể sắp xếp: {str(e)}")
                    
                    # Bước 6: Vẽ biểu đồ
                    st.subheader(f"📊 Biểu đồ: {', '.join(y_columns)} theo {x_column}")
                    
                    try:
                        if "Cột" in chart_type:
                            st.bar_chart(chart_data)
                            st.success("✅ Vẽ biểu đồ cột thành công!")
                        elif "Đường" in chart_type:
                            st.line_chart(chart_data)
                            st.success("✅ Vẽ biểu đồ đường thành công!")
                        elif "Vùng" in chart_type:
                            st.area_chart(chart_data)
                            st.success("✅ Vẽ biểu đồ vùng thành công!")
                        elif "Phân tán" in chart_type:
                            st.write("✅ Vẽ biểu đồ phân tán...")
                            fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
                            
                            # Xóa NaN trước khi vẽ scatter
                            df_scatter = df.dropna(subset=[x_column] + y_columns)
                            
                            if len(df_scatter) == 0:
                                st.error("❌ Không có dữ liệu hợp lệ để vẽ!")
                                st.stop()
                            
                            if df_scatter[x_column].dtype == 'object':
                                x_numeric = pd.factorize(df_scatter[x_column])[0]
                                x_label = x_column
                            else:
                                x_numeric = df_scatter[x_column]
                                x_label = x_column
                            
                            for y_col in y_columns:
                                ax.scatter(x_numeric, df_scatter[y_col], label=y_col, alpha=0.6, s=100)
                            
                            ax.set_xlabel(x_label)
                            ax.set_ylabel("Giá trị")
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            st.success("✅ Vẽ biểu đồ phân tán thành công!")
                    except Exception as e:
                        st.error(f"❌ Lỗi khi vẽ biểu đồ: {str(e)}")
                        st.write(f"**Chi tiết:** {type(e).__name__}")
                        st.stop()
                    
                    # Bước 7: Hiển thị dữ liệu
                    with st.expander("📊 Xem dữ liệu biểu đồ"):
                        st.write(f"**Kích thước:** {chart_data.shape[0]} dòng × {chart_data.shape[1]} cột")
                        st.dataframe(chart_data, width='stretch')
                        
                        # Tải CSV
                        csv_data = chart_data.to_csv(encoding='utf-8-sig')
                        st.download_button(
                            "⬇️ Tải dữ liệu biểu đồ (CSV)",
                            data=csv_data,
                            file_name=f"chart_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                except Exception as e:
                    st.error(f"❌ Lỗi không xác định: {str(e)}")
                    st.write(f"**Loại lỗi:** {type(e).__name__}")
                    st.write(f"**Dòng lỗi:** Kiểm tra console hoặc log")
    
    # --- TAB 3: THỐNG KÊ ---
    with tab3:
        st.header("Thống kê dữ liệu")
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if numeric_cols:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Tổng dòng", len(df))
            with col2:
                st.metric("📈 Cột số", len(numeric_cols))
            with col3:
                st.metric("📉 Cột chữ", df.shape[1] - len(numeric_cols))
            with col4:
                st.metric("🔢 Tổng cột", df.shape[1])
            
            st.write("**Thống kê chi tiết:**")
            st.dataframe(df.describe().T, width='stretch')
            
            # Ma trận tương quan
            if len(numeric_cols) > 1:
                with st.expander("🔗 Ma trận tương quan"):
                    corr_matrix = df[numeric_cols].corr()
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
                    st.pyplot(fig)
    
    # --- TAB 4: PHÂN TÍCH ---
    with tab4:
        st.header("Phân tích dữ liệu")
        
        col_a1, col_a2 = st.columns(2)
        
        with col_a1:
            st.write("**Phân bố dữ liệu theo cột:**")
            analyze_col = st.selectbox("Chọn cột để phân tích:", df.columns.tolist())
            
            if df[analyze_col].dtype in ['float64', 'int64']:
                fig, ax = plt.subplots()
                df[analyze_col].hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
                ax.set_title(f"Phân bố {analyze_col}")
                ax.set_xlabel("Giá trị")
                ax.set_ylabel("Tần số")
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots()
                df[analyze_col].value_counts().head(10).plot(kind='bar', ax=ax, color='coral')
                ax.set_title(f"Top 10 {analyze_col}")
                ax.set_xlabel("Giá trị")
                ax.set_ylabel("Số lượng")
                plt.xticks(rotation=45)
                st.pyplot(fig)
        
        with col_a2:
            st.write("**Kiểu dữ liệu từng cột:**")
            dtype_info = pd.DataFrame({
                'Cột': df.columns,
                'Kiểu': df.dtypes.astype(str),
                'Trống': df.isnull().sum()
            })
            st.dataframe(dtype_info, width='stretch')

else:
    st.info("📥 Chọn nguồn dữ liệu ở sidebar để bắt đầu phân tích")

# ===== FOOTER =====
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
    <p>📊 Dashboard Phân Tích Dữ Liệu v2.0 | Tạo bằng Streamlit | Cập nhật: 2025-12-21</p>
    </div>
    """,
    unsafe_allow_html=True
)
