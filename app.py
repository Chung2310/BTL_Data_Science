import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import io
from skimage import exposure
from scipy.ndimage import center_of_mass, shift

# Hàm tải và tiền xử lý dữ liệu
@st.cache_data
def load_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=True)
    X = mnist.data / 255.0  # Chuẩn hóa giá trị pixel
    y = mnist.target.astype(np.uint8)
    return X, y

# Hàm huấn luyện mô hình (được tối ưu hóa)
@st.cache_resource
def train_model(X_train, y_train, model_type):
    if model_type == "RandomForest":
        model = RandomForestClassifier(n_estimators=200, 
                                     max_depth=15,
                                     min_samples_split=5,
                                     random_state=42)
    else:
        model = SVC(kernel='rbf', 
                   C=10, 
                   gamma='scale', 
                   probability=True,
                   random_state=42)
    model.fit(X_train, y_train)
    return model

# Hàm vẽ confusion matrix chi tiết hơn
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
               annot_kws={"size": 10}, cbar=False)
    plt.title('Confusion Matrix', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    return fig

# Confusion matrix chuẩn hóa
def plot_normalized_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', ax=ax, fmt=".2f")
    plt.title("Normalized Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return fig

# Căn giữa chữ số
def center_image(img_array):
    cy, cx = center_of_mass(img_array)
    shift_y = np.round(img_array.shape[0]/2.0 - cy).astype(int)
    shift_x = np.round(img_array.shape[1]/2.0 - cx).astype(int)
    shifted = shift(img_array, shift=[shift_y, shift_x], mode='constant', cval=0.0)
    return shifted

# Hàm xử lý ảnh tải lên được cải tiến
def process_uploaded_image(uploaded_file):
    img = Image.open(uploaded_file).convert('L')
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(3.0)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    img = ImageOps.autocontrast(img, cutoff=5)
    img_array = np.array(img)
    img_array = exposure.equalize_hist(img_array)
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img) / 255.0
    img_array = 1 - img_array
    img_array = center_image(img_array)
    img_array = (img_array - img_array.mean()) / (img_array.std() + 1e-8)
    return img, img_array.flatten()

# Giao diện Streamlit được cải tiến

def main():
    st.set_page_config(page_title="MNIST Classifier", layout="wide")
    st.title("Phân loại chữ viết tay MNIST nâng cao")
    st.markdown("""
    <style>
    .reportview-container {
        background: #f0f2f6
    }
    .sidebar .sidebar-content {
        background: #ffffff
    }
    </style>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/2/2f/Google_2015_logo.svg", width=120)
        st.write("Một sản phẩm của Nhóm AI 🤖")
        st.markdown("---")
        st.header("Cài đặt mô hình")
        model_type = st.selectbox("Chọn mô hình:", ["RandomForest", "SVM"])
        show_details = st.checkbox("Hiển thị chi tiết mô hình", value=True)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with st.spinner("Đang huấn luyện mô hình..."):
        model = train_model(X_train, y_train, model_type)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Đánh giá mô hình")
        st.metric("Độ chính xác", f"{accuracy:.2%}")
        if show_details:
            st.text("Báo cáo phân loại:\n" + classification_report(y_test, y_pred))

    with col2:
        st.subheader("Confusion Matrix")
        st.pyplot(plot_confusion_matrix(y_test, y_pred))
        st.subheader("Normalized Confusion Matrix")
        st.pyplot(plot_normalized_confusion_matrix(y_test, y_pred))

    st.subheader("Dự đoán ảnh tải lên")
    st.markdown("""
    **Hướng dẫn sử dụng:**
    - Tải lên ảnh chứa **một chữ số duy nhất** (0-9)
    - Ảnh nên có nền trắng, chữ số màu đen
    - Chữ số nên nằm giữa ảnh và chiếm 60-80% diện tích
    """)

    uploaded_file = st.file_uploader("Chọn file ảnh...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            original_img, img_array = process_uploaded_image(uploaded_file)

            col1, col2 = st.columns(2)
            with col1:
                st.image(original_img, caption="Ảnh gốc", use_column_width=True)
            with col2:
                fig, ax = plt.subplots()
                ax.imshow(img_array.reshape(28, 28), cmap='gray')
                ax.axis('off')
                st.pyplot(fig)

            probs = model.predict_proba([img_array])[0]
            prediction = model.predict([img_array])[0]
            st.subheader("Kết quả dự đoán")
            st.metric("Dự đoán", str(prediction))

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(range(10), probs, color='skyblue')
            ax.set_xticks(range(10))
            ax.set_xlabel('Chữ số')
            ax.set_ylabel('Xác suất')
            ax.set_title('Phân phối xác suất dự đoán')
            ax.set_ylim(0, 1)
            st.pyplot(fig)

            top3 = np.argsort(probs)[-3:][::-1]
            st.write("Top 3 dự đoán:")
            for i, digit in enumerate(top3):
                st.write(f"{i+1}. Số {digit}: {probs[digit]:.2%}")

        except Exception as e:
            st.error(f"Lỗi khi xử lý ảnh: {str(e)}")
            st.info("Vui lòng tải lên ảnh hợp lệ theo hướng dẫn")

    st.subheader("Xem một số mẫu dự đoán")
    num_samples = st.slider("Số mẫu muốn xem", min_value=1, max_value=10, value=3)
    sample_indices = np.random.choice(len(X_test), num_samples, replace=False)
    for i, idx in enumerate(sample_indices):
        st.write(f"**Mẫu {i+1}**:")
        image = X_test.iloc[idx].values.reshape(28, 28)
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        st.pyplot(fig)
        st.write(f"Dự đoán: `{y_pred[idx]}`, Thực tế: `{y_test.iloc[idx]}`")

if __name__ == "__main__":
    main()
