import streamlit as st
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict_risk import predict_student
from src.intervention import suggest_interventions
from src.explainability import get_feature_importance
from src.evaluation import load_metrics
from utils.visualization import plot_feature_importance

st.set_page_config(page_title="StudentRisk AI", page_icon="🎓", layout="wide")
st.title("🎓 StudentRisk AI")
st.subheader("Early identification of students who may need academic support")
st.info(
    "Educational support prototype only. Predictions can be wrong and should never "
    "be used alone to label, punish, or make high-stakes decisions about a student."
)

model_path = ROOT / "models" / "best_model.pkl"
if not model_path.exists():
    st.warning("The model has not been trained yet. In the project folder, run: "
               "`python -m src.train_model`")
    st.stop()

st.markdown("### Enter early-stage student information")
st.caption("Use a fictional example for demonstration. Dataset feature codes are explained below.")

with st.form("student_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        school = st.selectbox("School", ["GP", "MS"])
        sex = st.selectbox("Sex", ["F", "M"])
        age = st.slider("Age", 15, 22, 17)
        address = st.selectbox("Home area", ["U", "R"], help="U = urban, R = rural")
        studytime = st.selectbox("Weekly study time", [1, 2, 3, 4],
            format_func=lambda x: {1:"Less than 2 hours",2:"2–5 hours",3:"5–10 hours",4:"More than 10 hours"}[x])
        failures = st.selectbox("Past class failures", [0, 1, 2, 3])
    with col2:
        absences = st.slider("Absences so far", 0, 30, 3)
        G1 = st.slider("First-period grade (0–20)", 0, 20, 10)
        schoolsup = st.selectbox("Extra school support", ["yes", "no"])
        famsup = st.selectbox("Family educational support", ["yes", "no"])
        paid = st.selectbox("Extra paid classes", ["yes", "no"])
        activities = st.selectbox("Extracurricular activities", ["yes", "no"])
    with col3:
        higher = st.selectbox("Wants higher education", ["yes", "no"])
        internet = st.selectbox("Internet access at home", ["yes", "no"])
        traveltime = st.selectbox("Travel time to school", [1, 2, 3, 4])
        health = st.slider("Self-reported health (1–5)", 1, 5, 3)
        famrel = st.slider("Family relationship quality (1–5)", 1, 5, 4)
        goout = st.slider("Going out with friends (1–5)", 1, 5, 3)
    submitted = st.form_submit_button("Predict academic risk")

if submitted:
    student = {
        "school": school, "sex": sex, "age": age, "address": address,
        "studytime": studytime, "failures": failures, "absences": absences,
        "G1": G1, "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
        "activities": activities, "higher": higher, "internet": internet,
        "traveltime": traveltime, "health": health, "famrel": famrel,
        "goout": goout
    }
    result = predict_student(student)
    if result["prediction"] == 1:
        st.error("Model result: At risk")
    else:
        st.success("Model result: Not identified as at risk")
    if result["risk_probability"] is not None:
        st.metric("Estimated model probability of at-risk class",
                  f"{result['risk_probability']:.1%}")
    st.markdown("#### Possible supportive next steps")
    for advice in suggest_interventions(student):
        st.write("•", advice)
    st.caption("This is a model estimate, not a certainty. Confirm context with the student and teaching staff.")

st.divider()
st.markdown("### Model evaluation")
try:
    metrics = load_metrics()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    c2.metric("At-risk precision", f"{metrics['precision_at_risk']:.1%}")
    c3.metric("At-risk recall", f"{metrics['recall_at_risk']:.1%}")
    c4.metric("At-risk F1", f"{metrics['f1_at_risk']:.1%}")
    st.caption("Metrics are calculated on a held-out test split. Performance on a new school may differ.")
except FileNotFoundError:
    st.info("Run model training to generate evaluation metrics.")

st.markdown("### Feature importance")
try:
    importance = get_feature_importance()
    st.pyplot(plot_feature_importance(importance))
except Exception as exc:
    st.caption(f"Feature importance unavailable: {exc}")

with st.expander("Dataset code guide"):
    st.markdown("""
    - **G1:** first-period grade; **G3:** final grade used only to create the training label.
    - **studytime:** 1 = under 2 hours/week, 2 = 2–5, 3 = 5–10, 4 = over 10.
    - **failures:** number of previous class failures.
    - **absences:** recorded absences in the dataset.
    - **U/R:** urban/rural; yes/no fields are categories.

    This public dataset is from Portuguese secondary schools. It is not a validated model
    for Indian engineering students. Use only for a student project and validate on
    suitable local data before considering real-world use.
    """)
