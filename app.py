import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cấu hình trang
st.set_page_config(page_title="Dashboard Tùy Chỉnh", layout="wide")
st.title("Phân Tích Dữ Liệu Tự Do 🛠️")
st.markdown("---")
url = "https://raw.githubusercontent.com/YOUR_USERNAME/python-data-analytics/main/data.csv"

# Đọc file
df = pd.read_csv(url)
# --- HÀM ĐỌC DỮ LIỆU ---
@st.cache_data
def load_data(uploaded_file):
    """Đọc file CSV hoặc Excel với xử lý lỗi"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, low_memory=False)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            return None
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {str(e)}")
        return None

# --- HÀM TỰ ĐỘNG CHUYỂN ĐỔI KIỂU DỮ LIỆU ---
def auto_convert_dtypes(df):
    """Chuyển đổi kiểu dữ liệu phù hợp cho các cột"""
    df_converted = df.copy()
    
    for col in df_converted.columns:
        if df_converted[col].dtype == 'object':
            # Xóa khoảng trắng đầu/cuối
            df_converted[col] = df_converted[col].astype(str).str.strip()
            
            # Thử chuyển sang số
            try:
                df_converted[col] = pd.to_numeric(
                    df_converted[col].str.replace(',', '', regex=False),
                    errors='coerce'
                )
            except:
                pass
            
            # Thử chuyển sang datetime
            if df_converted[col].dtype == 'object':
                try:
                    df_converted[col] = pd.to_datetime(
                        df_converted[col],
                        errors='coerce'
                    )
                except:
                    pass
    
    return df_converted

# --- UPLOAD FILE ---
st.sidebar.header("📁 Dữ liệu đầu vào")
uploaded_file = st.sidebar.file_uploader(
    "Upload file CSV/Excel", 
    type=['csv', 'xlsx', 'xls']
)

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.success(f"✅ Đã tải file: **{uploaded_file.name}**")
        st.info(f"📊 Kích thước: {df.shape[0]} dòng × {df.shape[1]} cột")
        
        # Tự động chuyển đổi kiểu dữ liệu
        df = auto_convert_dtypes(df)

        # Hiển thị bảng dữ liệu
        with st.expander("📋 Xem dữ liệu chi tiết"):
            col_preview, col_stats = st.columns(2)
            
            with col_preview:
                st.write("**Dữ liệu mẫu:**")
                st.dataframe(df.head(20), use_container_width=True)
            
            with col_stats:
                st.write("**Thống kê cơ bản:**")
                st.dataframe(df.describe(), use_container_width=True)

        # Xóa dòng trống
        df = df.dropna(how='all')

        # --- PHẦN TÙY CHỈNH BIỂU ĐỒ ---
        st.header("📈 Tùy chỉnh biểu đồ so sánh")
        
        col1, col2, col3 = st.columns(3)
        all_columns = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        with col1:
            x_column = st.selectbox(
                "Chọn Trục X (Hoành):", 
                all_columns,
                help="Trục ngang - thường là danh mục hoặc thời gian"
            )
            
        with col2:
            y_columns = st.multiselect(
                "Chọn Trục Y (Tung):", 
                numeric_cols,
                help="Trục dọc - chỉ hiện các cột số"
            )
            
        with col3:
            chart_type = st.selectbox(
                "Loại biểu đồ:", 
                ["Cột (Bar)", "Đường (Line)", "Vùng (Area)", "Phân tán (Scatter)"]
            )

        # Tùy chọn nâng cao
        with st.expander("⚙️ Tùy chọn nâng cao"):
            col_adv1, col_adv2 = st.columns(2)
            
            with col_adv1:
                use_groupby = st.checkbox(
                    "Gom nhóm dữ liệu", 
                    value=True,
                    help="Tính tổng theo nhóm X"
                )
                sort_ascending = st.checkbox("Sắp xếp tăng dần", value=True)
            
            with col_adv2:
                figsize_width = st.slider("Chiều rộng biểu đồ", 8, 16, 10)
                figsize_height = st.slider("Chiều cao biểu đồ", 4, 12, 6)

        if st.button("🚀 Vẽ biểu đồ", use_container_width=True):
            if len(y_columns) == 0:
                st.warning("⚠️ Vui lòng chọn ít nhất 1 cột SỐ cho Trục Y.")
            else:
                try:
                    st.subheader(f"Biểu đồ: {', '.join(y_columns)} theo {x_column}")
                    
                    # Xử lý dữ liệu
                    df_chart = df[[x_column] + y_columns].copy()
                    df_chart = df_chart.dropna(subset=y_columns)
                    
                    if use_groupby:
                        # Gom nhóm và tính tổng
                        chart_data = df_chart.groupby(x_column)[y_columns].sum()
                    else:
                        # Sắp xếp theo trục X
                        chart_data = df_chart.set_index(x_column)[y_columns]
                    
                    if sort_ascending:
                        chart_data = chart_data.sort_index()
                    
                    # Vẽ biểu đồ
                    if chart_type == "Cột (Bar)":
                        st.bar_chart(chart_data)
                    
                    elif chart_type == "Đường (Line)":
                        st.line_chart(chart_data)
                    
                    elif chart_type == "Vùng (Area)":
                        st.area_chart(chart_data)
                    
                    elif chart_type == "Phân tán (Scatter)":
                        fig, ax = plt.subplots(figsize=(figsize_width, figsize_height))
                        
                        # Nếu X là danh mục, chuyển sang số cho scatter
                        if df[x_column].dtype == 'object':
                            x_numeric = pd.factorize(df[x_column])[0]
                            x_label = "Nhóm"
                        else:
                            x_numeric = df[x_column]
                            x_label = x_column
                        
                        for y_col in y_columns:
                            ax.scatter(x_numeric, df[y_col], label=y_col, alpha=0.6, s=50)
                        
                        ax.set_xlabel(x_label)
                        ax.set_ylabel("Giá trị")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    # Hiển thị bảng dữ liệu biểu đồ
                    with st.expander("📊 Xem dữ liệu biểu đồ"):
                        st.dataframe(chart_data, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"❌ Lỗi khi vẽ: {str(e)}")
                    st.write("**Gợi ý:** Kiểm tra dữ liệu có giá trị trống hay không hợp lệ.")

        # --- THỐNG KÊ NHANH ---
        st.header("📊 Thống kê nhanh")
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        
        if numeric_cols:
            with stat_col1:
                st.metric("Tổng số dòng", len(df))
            with stat_col2:
                st.metric("Số cột", df.shape[1])
            with stat_col3:
                st.metric("Cột số", len(numeric_cols))
        
        # Tương quan giữa các cột
        if len(numeric_cols) > 1:
            with st.expander("🔗 Ma trận tương quan"):
                corr_matrix = df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
                st.pyplot(fig)

else:
    st.info("📥 Vui lòng upload file CSV hoặc Excel để bắt đầu phân tích dữ liệu.")
