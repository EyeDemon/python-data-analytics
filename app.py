import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình trang
st.set_page_config(page_title="Dashboard Tùy Chỉnh", layout="wide")
st.title("Phân Tích Dữ Liệu Tự Do 🛠️")
st.markdown("---")

# --- HÀM ĐỌC DỮ LIỆU ---
@st.cache_data
def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, low_memory=False)
        else:
            df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        return None

# --- UPLOAD FILE ---
st.sidebar.header("Dữ liệu đầu vào")
uploaded_file = st.sidebar.file_uploader("Upload file CSV/Excel", type=['csv', 'xlsx'])

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None:
        st.write(f"Đã tải file: **{uploaded_file.name}**")
        
        # --- BƯỚC MỚI: TỰ ĐỘNG CHUYỂN ĐỔI SỐ (FIX LỖI WARNING) ---
        # Tìm các cột có vẻ là số nhưng đang bị lưu là chữ
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Thử xóa dấu phẩy (1,000 -> 1000)
                    # errors='coerce': Nếu không chuyển được thành số thì biến thành NaN (trống)
                    # Đây là cách chuẩn mới, không gây Warning
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
                except:
                    pass

        # Hiển thị bảng dữ liệu
        with st.expander("Xem dữ liệu chi tiết"):
            st.dataframe(df.head(100))

        # --- PHẦN TÙY CHỈNH BIỂU ĐỒ ---
        st.header("Tùy chỉnh biểu đồ so sánh")
        
        col1, col2, col3 = st.columns(3)
        all_columns = df.columns.tolist()
        
        with col1:
            x_column = st.selectbox("Chọn Trục X (Hoành):", all_columns)
            
        with col2:
            # Chọn trục Y (Chỉ hiện các cột số thực sự)
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            y_columns = st.multiselect("Chọn Trục Y (Tung):", numeric_cols)
            
        with col3:
            chart_type = st.selectbox("Loại biểu đồ:", ["Đường (Line)", "Cột (Bar)", "Vùng (Area)", "Phân tán (Scatter)"])

        if st.button("Vẽ biểu đồ ngay 🚀"):
            if len(y_columns) > 0:
                st.subheader(f"Biểu đồ: {', '.join(y_columns)} theo {x_column}")
                
                try:
                    # Logic xử lý dữ liệu
                    if df[x_column].dtype == 'object' or len(df[x_column].unique()) < len(df)/2:
                        # Gom nhóm và tính tổng
                        chart_data = df.groupby(x_column)[y_columns].sum()
                    else:
                        # Sắp xếp theo trục X
                        chart_data = df.set_index(x_column)[y_columns].sort_index()

                    # Vẽ biểu đồ
                    if chart_type == "Cột (Bar)":
                        st.bar_chart(chart_data)
                    elif chart_type == "Đường (Line)":
                        st.line_chart(chart_data)
                    elif chart_type == "Vùng (Area)":
                        st.area_chart(chart_data)
                    elif chart_type == "Phân tán (Scatter)":
                        fig, ax = plt.subplots(figsize=(10, 6))
                        for y_col in y_columns:
                            sns.scatterplot(data=df, x=x_column, y=y_col, label=y_col, ax=ax)
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"Lỗi khi vẽ: {e}")
            else:
                st.warning("Vui lòng chọn ít nhất 1 cột SỐ cho Trục Y.")
    else:
        st.error("Lỗi đọc file.")
else:
    st.info("Vui lòng upload file để bắt đầu.")
