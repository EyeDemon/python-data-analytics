import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình trang web rộng để hiển thị bảng lớn dễ hơn
st.set_page_config(page_title="Dashboard Phân Tích Dữ Liệu Lớn", layout="wide")

st.title("Phân Tích Dữ Liệu - Big Data Mode 🚀")
st.markdown("---")

# --- KỸ THUẬT 1: HÀM ĐỌC DỮ LIỆU CÓ CACHING ---
# Hàm này giúp lưu dữ liệu vào bộ nhớ đệm, không cần load lại khi tương tác
@st.cache_data
def load_data(uploaded_file):
    try:
        # Nếu là file CSV
        if uploaded_file.name.endswith('.csv'):
            # Dùng low_memory=False để xử lý các cột hỗn hợp kiểu dữ liệu
            df = pd.read_csv(uploaded_file, low_memory=False)
        # Nếu là file Excel
        else:
            df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        return None

# --- GIAO DIỆN UPLOAD ---
sidebar = st.sidebar
sidebar.header("Khu vực Upload")
uploaded_file = sidebar.file_uploader("Chọn file dữ liệu lớn (CSV/Excel)", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Gọi hàm đọc dữ liệu thông minh
    with st.spinner('Đang xử lý dữ liệu lớn... xin vui lòng chờ...'):
        df = load_data(uploaded_file)

    if df is not None:
        # --- KỸ THUẬT 2: HIỂN THỊ THÔNG MINH ---
        # Chỉ hiện thông tin tổng quan để tránh lag
        row_count = df.shape[0]
        col_count = df.shape[1]
        
        st.success(f"✅ Đã tải thành công! Kích thước: {row_count:,} dòng, {col_count} cột.")
        
        # Xem trước dữ liệu (Giới hạn hiển thị để mượt mà)
        st.subheader("1. Xem trước dữ liệu")
        with st.expander("Bấm để xem bảng dữ liệu chi tiết"):
            if row_count > 1000:
                st.warning("⚠️ File quá lớn, chỉ hiển thị 1000 dòng đầu tiên để tối ưu hiệu năng.")
                st.dataframe(df.head(1000))
            else:
                st.dataframe(df)

        # --- PHẦN PHÂN TÍCH TỰ ĐỘNG ---
        st.subheader("2. Thống kê & Biểu đồ")
        
        # Tự động lọc ra các cột SỐ và cột CHỮ
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        col1, col2 = st.columns([1, 2])

        with col1:
            st.info("Tùy chọn vẽ biểu đồ")
            # Chọn cột để phân tích
            if object_cols:
                cat_col = st.selectbox("Chọn cột phân nhóm (Trục X):", object_cols)
            else:
                cat_col = None
                
            if numeric_cols:
                num_col = st.selectbox("Chọn cột giá trị (Trục Y):", numeric_cols)
                chart_type = st.radio("Loại biểu đồ:", ["Cột (Bar)", "Đường (Line)", "Tròn (Pie)"])
            else:
                num_col = None

        with col2:
            if cat_col and num_col:
                st.markdown(f"**Biểu đồ thể hiện: {num_col} theo {cat_col}**")
                
                # Gom nhóm dữ liệu (Groupby) - Bước quan trọng để xử lý file lớn
                # Thay vì vẽ 10.000 điểm, ta chỉ vẽ kết quả tổng hợp
                df_grouped = df.groupby(cat_col)[num_col].sum().sort_values(ascending=False).head(15) # Chỉ lấy Top 15 để vẽ cho đẹp
                
                fig, ax = plt.subplots(figsize=(10, 5))
                
                if chart_type == "Cột (Bar)":
                    sns.barplot(x=df_grouped.values, y=df_grouped.index, ax=ax, palette="viridis")
                    ax.set_xlabel(num_col)
                elif chart_type == "Đường (Line)":
                    df_grouped.plot(kind='line', marker='o', ax=ax)
                elif chart_type == "Tròn (Pie)":
                    df_grouped.plot.pie(autopct='%1.1f%%', ax=ax)
                    ax.set_ylabel('')
                
                st.pyplot(fig)
            else:
                st.warning("Dữ liệu không đủ điều kiện để vẽ (cần ít nhất 1 cột chữ và 1 cột số).")

    else:
        st.error("File bị lỗi hoặc không đọc được. Hãy kiểm tra lại định dạng.")

else:
    st.info("Chưa có file nào được chọn. Hãy upload file ở cột bên trái.")
