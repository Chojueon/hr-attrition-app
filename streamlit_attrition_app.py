import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(
    page_title="조기퇴사 예측 테스트",
    page_icon="📊",
    layout="wide"
)

DATA_PATH = "HR_DATA.csv"

DROP_COLS = [
    "EmployeeCount",
    "EmployeeNumber",
    "Over18",
    "StandardHours",
]

LEAKAGE_COLS = [
    # 입사 초기 예측 도구로 쓰려면 장기 근속 이후에만 알 수 있는 변수는 제외하는 것이 안전합니다.
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource
def train_model(df: pd.DataFrame):
    work_df = df.copy()
    work_df["AttritionFlag"] = work_df["Attrition"].map({"No": 0, "Yes": 1})

    cols_to_drop = [c for c in DROP_COLS + LEAKAGE_COLS + ["Attrition"] if c in work_df.columns]
    work_df = work_df.drop(columns=cols_to_drop)

    X = work_df.drop(columns=["AttritionFlag"])
    y = work_df["AttritionFlag"]

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        max_depth=8,
        min_samples_leaf=5,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "features": X.columns.tolist(),
    }

    return pipeline, metrics, X


def risk_label(prob: float) -> str:
    if prob >= 0.70:
        return "High Risk"
    if prob >= 0.40:
        return "Medium Risk"
    return "Low Risk"


def risk_comment(prob: float) -> str:
    if prob >= 0.70:
        return "조기퇴사 가능성이 높은 편입니다. 온보딩, 업무강도, 관리자 면담 등 즉각적인 리텐션 액션을 권장합니다."
    if prob >= 0.40:
        return "조기퇴사 가능성이 중간 수준입니다. 초기 적응 상태와 업무 만족도를 점검하는 것이 좋습니다."
    return "현재 입력값 기준 조기퇴사 가능성은 낮은 편입니다. 단, 실제 판단에는 면담과 정성 정보가 함께 필요합니다."


st.title("📊 HR 조기퇴사 예측 테스트")
st.caption("Kaggle HR 데이터를 기반으로 학습한 모델을 사용해 입력 직원의 퇴사 가능성을 예측합니다.")

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error("HR_DATA.csv 파일을 app.py와 같은 폴더에 넣어주세요.")
    st.stop()

pipeline, metrics, X_template = train_model(df)

with st.expander("모델 정보 보기"):
    col1, col2, col3 = st.columns(3)
    col1.metric("Train 데이터", f"{metrics['train_shape'][0]:,}명")
    col2.metric("Test 데이터", f"{metrics['test_shape'][0]:,}명")
    col3.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    st.info("입사 초기 예측 도구를 가정했기 때문에 YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, YearsWithCurrManager 변수는 데이터 누수 방지를 위해 제외했습니다.")

st.subheader("1. 직원 정보 입력")

left, right = st.columns(2)

with left:
    age = st.slider("나이", 18, 65, 30)
    gender = st.selectbox("성별", sorted(df["Gender"].dropna().unique()))
    marital_status = st.selectbox("결혼 여부", sorted(df["MaritalStatus"].dropna().unique()))
    education = st.selectbox("학력 수준", sorted(df["Education"].dropna().unique()))
    education_field = st.selectbox("전공", sorted(df["EducationField"].dropna().unique()))
    distance_from_home = st.slider("집과 회사 거리", 1, 30, 10)
    num_companies_worked = st.slider("이전 근무 회사 수", 0, 10, 2)
    total_working_years = st.slider("총 경력 연수", 0, 40, 5)

with right:
    department = st.selectbox("부서", sorted(df["Department"].dropna().unique()))
    job_role = st.selectbox("직무", sorted(df["JobRole"].dropna().unique()))
    job_level = st.selectbox("직급 레벨", sorted(df["JobLevel"].dropna().unique()))
    business_travel = st.selectbox("출장 빈도", sorted(df["BusinessTravel"].dropna().unique()))
    overtime = st.selectbox("초과근무 여부", sorted(df["OverTime"].dropna().unique()))
    monthly_income = st.number_input("월 소득", min_value=500, max_value=25000, value=5000, step=100)
    percent_salary_hike = st.slider("최근 임금 인상률(%)", 10, 30, 15)
    stock_option_level = st.selectbox("스톡옵션 레벨", sorted(df["StockOptionLevel"].dropna().unique()))

st.subheader("2. 근무 만족도 및 평가 정보")

c1, c2, c3, c4 = st.columns(4)
with c1:
    environment_satisfaction = st.slider("근무환경 만족도", 1, 4, 3)
    job_satisfaction = st.slider("직무 만족도", 1, 4, 3)
with c2:
    relationship_satisfaction = st.slider("관계 만족도", 1, 4, 3)
    work_life_balance = st.slider("워라밸", 1, 4, 3)
with c3:
    job_involvement = st.slider("직무 몰입도", 1, 4, 3)
    performance_rating = st.slider("성과 등급", 3, 4, 3)
with c4:
    training_times_last_year = st.slider("최근 1년 교육 횟수", 0, 6, 2)

# 모델 학습에 포함된 나머지 수치형 변수는 데이터 중앙값으로 채웁니다.
input_data = {}
for col in X_template.columns:
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        input_data[col] = float(df[col].median())
    else:
        input_data[col] = df[col].mode()[0] if col in df.columns else None

input_data.update({
    "Age": age,
    "BusinessTravel": business_travel,
    "Department": department,
    "DistanceFromHome": distance_from_home,
    "Education": education,
    "EducationField": education_field,
    "EnvironmentSatisfaction": environment_satisfaction,
    "Gender": gender,
    "JobInvolvement": job_involvement,
    "JobLevel": job_level,
    "JobRole": job_role,
    "JobSatisfaction": job_satisfaction,
    "MaritalStatus": marital_status,
    "MonthlyIncome": monthly_income,
    "NumCompaniesWorked": num_companies_worked,
    "OverTime": overtime,
    "PercentSalaryHike": percent_salary_hike,
    "PerformanceRating": performance_rating,
    "RelationshipSatisfaction": relationship_satisfaction,
    "StockOptionLevel": stock_option_level,
    "TotalWorkingYears": total_working_years,
    "TrainingTimesLastYear": training_times_last_year,
    "WorkLifeBalance": work_life_balance,
})

input_df = pd.DataFrame([input_data])[X_template.columns]

st.divider()

if st.button("조기퇴사 가능성 예측하기", type="primary"):
    prob = float(pipeline.predict_proba(input_df)[0][1])
    pred = int(prob >= 0.5)
    label = risk_label(prob)

    st.subheader("3. 예측 결과")

    r1, r2, r3 = st.columns(3)
    r1.metric("퇴사 확률", f"{prob * 100:.1f}%")
    r2.metric("위험 등급", label)
    r3.metric("예측 라벨", "퇴사 가능" if pred == 1 else "재직 가능")

    st.progress(min(prob, 1.0))
    st.warning(risk_comment(prob)) if prob >= 0.7 else st.info(risk_comment(prob))

    st.subheader("입력값 확인")
    st.dataframe(input_df.T.rename(columns={0: "입력값"}))

st.caption("주의: 이 앱은 Kaggle 예제 데이터 기반 테스트 도구입니다. 실제 인사 의사결정에는 조직별 실제 라벨 데이터, 법적·윤리적 검토, 편향성 점검이 필요합니다.")
