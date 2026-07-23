import streamlit as st

from predict import AQ_QUESTIONS, load_artifacts, predict

st.set_page_config(
    page_title="Autism Screening Predictor",
    page_icon="🧠",
    layout="wide",
)

st.title("Autism Screening Predictor")
st.caption(
    "Answer the AQ-10 questionnaire and demographic questions to receive a "
    "machine-learning screening result."
)

st.warning(
    "This tool is for educational purposes only and is not a medical diagnosis. "
    "Please consult a qualified healthcare professional for clinical assessment."
)


@st.cache_resource
def get_model():
    return load_artifacts()


model, encoders, lookup = get_model()

with st.form("screening_form"):
    st.subheader("AQ-10 Questionnaire")
    st.markdown("For each statement, select **Yes** if it applies to you, otherwise **No**.")

    scores = {}
    cols = st.columns(2)
    for idx, (key, question) in enumerate(AQ_QUESTIONS.items()):
        with cols[idx % 2]:
            scores[key] = st.radio(question, options=[0, 1], format_func=lambda v: "Yes" if v else "No", key=key)

    st.subheader("Demographics")
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age (years)", min_value=1.0, max_value=100.0, value=25.0, step=0.1)
        gender = st.selectbox("Gender", options=list(encoders["gender"].classes_))
        ethnicity = st.selectbox("Ethnicity", options=list(encoders["ethnicity"].classes_))

    with c2:
        jaundice = st.selectbox("Had jaundice at birth?", options=list(encoders["jaundice"].classes_))
        austim = st.selectbox(
            "Family member with autism?",
            options=list(encoders["austim"].classes_),
        )
        relation = st.selectbox("Who is completing this form?", options=list(encoders["relation"].classes_))

    with c3:
        country = st.selectbox("Country of residence", options=sorted(encoders["contry_of_res"].classes_))
        used_app_before = st.selectbox(
            "Used this app before?",
            options=list(encoders["used_app_before"].classes_),
        )

    submitted = st.form_submit_button("Get Prediction", type="primary")

if submitted:
    payload = {
        **scores,
        "age": age,
        "gender": gender,
        "ethnicity": ethnicity,
        "jaundice": jaundice,
        "austim": austim,
        "contry_of_res": country,
        "used_app_before": used_app_before,
        "relation": relation,
    }

    label, confidence = predict(payload, model, encoders, lookup)

    st.divider()
    if label == 1:
        st.error(f"**Screening result: ASD traits likely** (confidence {confidence:.0%})")
    else:
        st.success(f"**Screening result: No ASD traits detected** (confidence {confidence:.0%})")

    st.info(
        "The model is a Random Forest classifier trained on AQ-10 screening data. "
        "Use this result only as an informational signal."
    )
