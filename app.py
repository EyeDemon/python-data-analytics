import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình trang
st.set_page_config(page_title="Dashboard Tùy Chỉnh", layout="wide")
st.title("Phân Tích Dữ Liệu Tự Do 🛠️")
st.markdown("---")

# --- HÀM ĐỌC DỮ LIỆU (Giữ nguyên để chạy nhanh) ---
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
    # Load data
    df = load_data(uploaded_file)
    
    if df is not None:
        st.write(f"Đã tải file: **{uploaded_file.name}** ({df.shape[0]} dòng)")
        
        # Hiện bảng dữ liệu (trong Expander cho gọn)
        with st.expander("Xem dữ liệu chi tiết"):
            st.dataframe(df.head(1000))

        # --- PHẦN TÙY CHỈNH BIỂU ĐỒ (MỚI) ---
        st.header("Tùy chỉnh biểu đồ so sánh")
        
        # Chia cột để chọn thông số
        col1, col2, col3 = st.columns(3)
        
        all_columns = df.columns.tolist()
        
        with col1:
            # Chọn trục X (Chỉ 1 cột)
            x_column = st.selectbox("Chọn Trục X (Hoành):", all_columns)
            
        with col2:
            # Chọn trục Y (Nhiều cột)
            y_columns = st.multiselect("Chọn Trục Y (Tung):", all_columns)
            
        with col3:
            # Chọn loại biểu đồ
            chart_type = st.selectbox("Loại biểu đồ:", ["Đường (Line)", "Cột (Bar)", "Phân tán (Scatter)", "Vùng (Area)"])

        # Nút vẽ biểu đồ
        if st.button("Vẽ biểu đồ ngay 🚀"):
            if len(y_columns) > 0:
                try:
                    # Tạo khung vẽ
                    st.subheader(f"Biểu đồ: {', '.join(y_columns)} theo {x_column}")
                    
                    # --- XỬ LÝ DỮ LIỆU TRƯỚC KHI VẼ ---
                    # Nếu trục X là dạng chữ (ví dụ Tên Sản Phẩm), ta cần gom nhóm (Group By)
                    # Nếu trục X là dạng số/ngày (ví dụ Nhiệt độ), ta vẽ trực tiếp
                    
                    # Kiểm tra xem X có nhiều giá trị trùng lặp không (để quyết định gom nhóm)
                    if df[x_column].dtype == 'object' or len(df[x_column].unique()) < len(df):
                        # Tự động tính tổng (Sum) cho các cột Y được chọn theo X
                        chart_data = df.groupby(x_column)[y_columns].sum()
                    else:
                        # Dữ liệu dạng liên tục, set index là X để vẽ
                        chart_data = df.set_index(x_column)[y_columns]

                    # --- VẼ BIỂU ĐỒ ---
                    if chart_type == "Cột (Bar)":
                        st.bar_chart(chart_data)
                    elif chart_type == "Đường (Line)":
                        st.line_chart(chart_data)
                    elif chart_type == "Vùng (Area)":
                        st.area_chart(chart_data)
                    elif chart_type == "Phân tán (Scatter)":
                        # Scatter cần vẽ bằng Matplotlib/Seaborn vì Streamlit basic không hỗ trợ tốt scatter đa biến
                        fig, ax = plt.subplots(figsize=(10, 6))
                        for y_col in y_columns:
                            sns.scatterplot(data=df, x=x_column, y=y_col, label=y_col, ax=ax)
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"Không thể vẽ biểu đồ: {e}. \n(Gợi ý: Hãy đảm bảo Trục Y là cột SỐ).")
            else:
                st.warning("Vui lòng chọn ít nhất 1 cột cho Trục Y.")
    else:
        st.error("Lỗi đọc file.")
else:
    st.info("Vui lòng upload file để bắt đầu.")
