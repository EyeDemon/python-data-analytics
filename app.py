import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Tiêu đề
st.title("Phân Tích Dữ Liệu Bán Hàng 📊")
st.write("Chào mừng bạn đến với ứng dụng phân tích dữ liệu cơ bản.")

# 2. Upload file
uploaded_file = st.file_uploader("Chọn file dữ liệu (CSV/Excel)", type=['csv', 'xlsx'])

# 3. Logic xử lý khi có file
if uploaded_file is not None:
    try:
        # Đọc file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("Tải file thành công!")

        # Xem dữ liệu
        st.subheader("Dữ liệu của bạn:")
        st.dataframe(df.head())

        # Vẽ biểu đồ đơn giản (Demo)
        st.subheader("Biểu đồ phân phối:")
        # Lấy cột số đầu tiên tìm thấy để vẽ
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            column_to_plot = st.selectbox("Chọn cột để vẽ biểu đồ:", numeric_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[column_to_plot], kde=True, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("File không có cột số để vẽ biểu đồ.")

    except Exception as e:
        st.error(f"Lỗi: {e}")
