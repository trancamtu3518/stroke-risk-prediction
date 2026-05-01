import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Thiết lập cấu hình trang
st.set_page_config(
    page_title="Dự đoán Đột quỵ",
    page_icon="🏥",
    layout="wide"
)

MODEL_PATH = Path("models") / "final_voting_clf.joblib"
PIPELINE_PATH = Path("models") / "pipeline_knn.joblib"

# Mapping từ tiếng Việt sang tiếng Anh
GENDER_MAPPING = {
    "Nam": "Male",
    "Nữ": "Female",
    "Khác": "Other"
}

MARRIED_MAPPING = {
    "Có": "Yes",
    "Không": "No"
}

WORK_TYPE_MAPPING = {
    "Tư nhân": "Private",
    "Tự kinh doanh": "Self-employed",
    "Công việc chính phủ": "Govt_job",
    "Trẻ em": "children",
    "Chưa bao giờ làm việc": "Never_worked"
}

RESIDENCE_MAPPING = {
    "Thành thị": "Urban",
    "Nông thôn": "Rural"
}

SMOKING_MAPPING = {
    "Chưa bao giờ hút thuốc": "never smoked",
    "Đã từng hút thuốc": "formerly smoked",
    "Đang hút thuốc": "smokes",
    "Không rõ": "Unknown"
}

@st.cache_resource
def load_model_and_pipeline():
    """Tải model và pipeline"""
    try:
        model = joblib.load(MODEL_PATH)
        pipeline = joblib.load(PIPELINE_PATH)
        return model, pipeline
    except Exception as e:
        st.error(f"Lỗi khi tải model hoặc pipeline: {e}")
        return None, None

def main():
    # Tiêu đề ứng dụng
    st.title("🏥 Hệ thống Dự đoán Nguy cơ Đột quỵ")
    st.markdown("---")
    
    # Tải model và pipeline
    model, pipeline = load_model_and_pipeline()
    
    if model is None or pipeline is None:
        st.error("Không thể tải model hoặc pipeline. Vui lòng kiểm tra lại đường dẫn file.")
        return
    
    st.success("✅ Model và pipeline đã được tải thành công!")
    
    # Tạo form nhập liệu
    st.header("📋 Nhập thông tin bệnh nhân")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Thông tin cá nhân")
        
        # Gender
        gender_vi = st.selectbox(
            "Giới tính",
            options=list(GENDER_MAPPING.keys()),
            help="Chọn giới tính của bệnh nhân"
        )
        
        # Age
        age = st.number_input(
            "Tuổi",
            min_value=0,
            max_value=120,
            value=45,
            step=1,
            help="Nhập tuổi của bệnh nhân (từ 0 đến 120)"
        )
        
        # Ever married
        ever_married_vi = st.selectbox(
            "Tình trạng hôn nhân",
            options=list(MARRIED_MAPPING.keys()),
            help="Bệnh nhân đã từng kết hôn chưa?"
        )
        
        # Work type
        work_type_vi = st.selectbox(
            "Loại công việc",
            options=list(WORK_TYPE_MAPPING.keys()),
            help="Chọn loại công việc của bệnh nhân"
        )
        
        # Residence type
        residence_type_vi = st.selectbox(
            "Loại nơi cư trú",
            options=list(RESIDENCE_MAPPING.keys()),
            help="Bệnh nhân sống ở thành thị hay nông thôn?"
        )
    
    with col2:
        st.subheader("Thông tin sức khỏe")
        
        # Hypertension
        hypertension = st.selectbox(
            "Tình trạng tăng huyết áp",
            options=[0, 1],
            format_func=lambda x: "Có" if x == 1 else "Không",
            help="Bệnh nhân có bị tăng huyết áp không? (0: Không, 1: Có)"
        )
        
        # Heart disease
        heart_disease = st.selectbox(
            "Bệnh tim",
            options=[0, 1],
            format_func=lambda x: "Có" if x == 1 else "Không",
            help="Bệnh nhân có bị bệnh tim không? (0: Không, 1: Có)"
        )
        
        # Average glucose level
        avg_glucose_level = st.number_input(
            "Mức glucose trung bình",
            min_value=0.0,
            max_value=500.0,
            value=100.0,
            step=0.1,
            help="Nhập mức glucose trung bình trong máu (mg/dL)"
        )
        
        # BMI
        bmi = st.number_input(
            "Chỉ số BMI",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=0.1,
            help="Nhập chỉ số khối cơ thể (Body Mass Index)"
        )
        
        # Smoking status
        smoking_status_vi = st.selectbox(
            "Tình trạng hút thuốc",
            options=list(SMOKING_MAPPING.keys()),
            help="Chọn tình trạng hút thuốc của bệnh nhân"
        )
    
    st.markdown("---")
    
    # Nút dự đoán
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        predict_button = st.button("🔍 Dự đoán Nguy cơ Đột quỵ", use_container_width=True, type="primary")
    
    # Xử lý khi nhấn nút dự đoán
    if predict_button:
        try:
            # Chuyển đổi từ tiếng Việt sang tiếng Anh
            gender = GENDER_MAPPING[gender_vi]
            ever_married = MARRIED_MAPPING[ever_married_vi]
            work_type = WORK_TYPE_MAPPING[work_type_vi]
            Residence_type = RESIDENCE_MAPPING[residence_type_vi]
            smoking_status = SMOKING_MAPPING[smoking_status_vi]
            
            # Tạo DataFrame từ dữ liệu nhập vào
            input_data = pd.DataFrame({
                'gender': [gender],
                'age': [age],
                'hypertension': [hypertension],
                'heart_disease': [heart_disease],
                'ever_married': [ever_married],
                'work_type': [work_type],
                'Residence_type': [Residence_type],
                'avg_glucose_level': [avg_glucose_level],
                'bmi': [bmi],
                'smoking_status': [smoking_status]
            })
            
            # Transform dữ liệu bằng pipeline
            transformed_data = pipeline.transform(input_data)
            
            # Dự đoán
            prediction = model.predict(transformed_data)
            prediction_proba = model.predict_proba(transformed_data)
            
            # Hiển thị kết quả
            st.markdown("---")
            st.header("📊 Kết quả Dự đoán")
            
            # Tạo 3 cột để hiển thị kết quả đẹp hơn
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col2:
                if prediction[0] == 0:
                    st.success("### ✅ KHÔNG CÓ NGUY CƠ ĐỘT QUỴ")
                    st.balloons()
                    st.info(f"**Xác suất không bị đột quỵ:** {prediction_proba[0][0]*100:.2f}%")
                else:
                    st.error("### ⚠️ CÓ NGUY CƠ BỊ ĐỘT QUỴ")
                    st.warning("Khuyến nghị: Nên đi khám bác sĩ để được tư vấn và kiểm tra sức khỏe!")
                    st.info(f"**Xác suất bị đột quỵ:** {prediction_proba[0][1]*100:.2f}%")
            
            # Hiển thị thông tin chi tiết
            with st.expander("📋 Xem thông tin chi tiết đã nhập"):
                st.write("**Thông tin bệnh nhân:**")
                st.dataframe(input_data, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Lỗi khi thực hiện dự đoán: {e}")
            st.exception(e)

if __name__ == "__main__":
    main()
