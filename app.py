import streamlit as st
import pandas as pd
import joblib
import time
import plotly.express as px

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import io

def create_pdf(skills, experience, education, certification, job_role,
               salary, projects, ai_score, result, confidence):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    story = []

    story.append(Paragraph("<b>AI Hiring Prediction Report</b>", styles["Title"]))

    story.append(Paragraph(f"<b>Skills:</b> {skills}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Experience:</b> {experience} Years", styles["BodyText"]))
    story.append(Paragraph(f"<b>Education:</b> {education}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Certification:</b> {certification}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Job Role:</b> {job_role}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Salary Expectation:</b> ${salary}", styles["BodyText"]))
    story.append(Paragraph(f"<b>Projects:</b> {projects}", styles["BodyText"]))
    story.append(Paragraph(f"<b>AI Score:</b> {ai_score}", styles["BodyText"]))

    story.append(Paragraph("<br/>", styles["BodyText"]))

    story.append(Paragraph(f"<b>Prediction:</b> {result}", styles["Heading2"]))
    story.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles["Heading2"]))

    doc.build(story)

    buffer.seek(0)

    return buffer

# =====================================
# Load Model, Encoders and Dataset
# =====================================
model = joblib.load("hiring_prediction_model.pkl")
encoders = joblib.load("label_encoders.pkl")
df = pd.read_csv("resume_dataset.csv")

# =====================================
# Page Configuration
# =====================================
st.set_page_config(
    page_title="AI Hiring Prediction System",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

h1, h2, h3{
    color:#1f77b4;
}

div[data-testid="stMetric"]{
    background-color:#ffffff;
    border-radius:12px;
    padding:15px;
    box-shadow:0px 3px 10px rgba(0,0,0,0.1);
}

/* Force metric text to stay black */
div[data-testid="stMetric"] *{
    color:#000000 !important;
}

/* Button */
div.stButton > button{
    background:#1f77b4;
    color:white;
    border-radius:12px;
    height:55px;
    font-size:20px;
    font-weight:bold;
}

div.stButton > button:hover{
    background:#125a9c;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# Sidebar
# =====================================
st.sidebar.title("🤖 AI Hiring System")

st.sidebar.info("""
This application predicts whether a candidate is likely to be hired based on:

• Skills
• Experience
• Education
• Certifications
• Job Role
• Salary Expectation
• Projects Count
• AI Score
""")

st.sidebar.success("Machine Learning Model: Random Forest")

# =====================================
# Title
# =====================================
st.markdown("""
# 🤖 AI Hiring Prediction System

### Intelligent Resume Screening using Machine Learning

---
""")

st.info("Fill in the candidate details below and click **Predict Hiring Decision**.")

st.markdown("## 📈 Dashboard Overview")

c1,c2,c3=st.columns(3)

with c1:
    st.metric("📄 Total Candidates",len(df))

with c2:
    st.metric("🤖 ML Model","Random Forest")

with c3:
    st.metric("🎯 Features",8)

st.markdown("---")

st.markdown("---")

# =====================================
# Dropdown Options
# =====================================
skills_options = sorted(df["Skills"].dropna().unique())
education_options = sorted(df["Education"].dropna().unique())
certification_options = sorted(df["Certifications"].dropna().unique())
job_role_options = sorted(df["Job Role"].dropna().unique())

# =====================================
# Input Fields
# =====================================

col1, col2 = st.columns(2)

with col1:
    skills = st.selectbox("💻 Skills", skills_options)

    experience = st.number_input(
        "📅 Experience (Years)",
        min_value=0,
        max_value=30,
        value=1
    )

    education = st.selectbox(
        "🎓 Education",
        education_options
    )

    certification = st.selectbox(
        "📜 Certifications",
        certification_options
    )

with col2:
    job_role = st.selectbox(
        "💼 Job Role",
        job_role_options
    )

    salary = st.number_input(
        "💲 Salary Expectation ($)",
        min_value=0,
        value=50000
    )

    projects = st.number_input(
        "📂 Projects Count",
        min_value=0,
        value=1
    )

    ai_score = st.slider(
        "🤖 AI Score (0-100)",
        min_value=0,
        max_value=100,
        value=70
    )

st.markdown("---")

# =====================================
# Candidate Details
# =====================================

st.markdown("## 📊 Candidate Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Experience", f"{experience} Years")
col2.metric("Projects", projects)
col3.metric("AI Score", ai_score)
col4.metric("Salary", f"${salary}")


st.markdown("## 👤 Candidate Summary")

st.info(f"""
### Candidate Information

**💻 Skills:** {skills}

**📅 Experience:** {experience} Years

**🎓 Education:** {education}

**📜 Certification:** {certification}

**💼 Job Role:** {job_role}

**💲 Salary Expectation:** ${salary}

**📂 Projects:** {projects}

**🤖 AI Score:** {ai_score}
""")

st.markdown("---")

# =====================================
# Prediction Button
# =====================================
if st.button("🚀 Predict Hiring Decision", use_container_width=True):

    with st.spinner("🔍 Analyzing Resume..."):
        time.sleep(2)

    # Encode categorical inputs
    skills_encoded = encoders["Skills"].transform([skills])[0]
    education_encoded = encoders["Education"].transform([education])[0]
    certification_encoded = encoders["Certifications"].transform([certification])[0]
    job_role_encoded = encoders["Job Role"].transform([job_role])[0]

    # Create input dataframe
    input_data = pd.DataFrame({
        "Skills": [skills_encoded],
        "Experience (Years)": [experience],
        "Education": [education_encoded],
        "Certifications": [certification_encoded],
        "Job Role": [job_role_encoded],
        "Salary Expectation ($)": [salary],
        "Projects Count": [projects],
        "AI Score (0-100)": [ai_score]
    })

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.subheader("🎯 Prediction Result")

    # ----------------------------
    # HIRED
    # ----------------------------
    if prediction == 1:

        confidence = probability[1] * 100
        result = "HIRED"

        st.success("✅ Candidate is likely to be HIRED")
        st.write(f"Confidence: **{confidence:.2f}%**")
        st.progress(float(probability[1]))

        st.subheader("⭐ Candidate Rating")

        if confidence >= 90:
            st.success("⭐⭐⭐⭐⭐ Excellent Candidate")
        elif confidence >= 75:
            st.info("⭐⭐⭐⭐ Good Candidate")
        elif confidence >= 60:
            st.warning("⭐⭐⭐ Average Candidate")
        else:
            st.warning("⭐⭐ Needs Improvement")

    # ----------------------------
    # REJECTED
    # ----------------------------
    else:

        confidence = probability[0] * 100
        result = "REJECTED"

        st.error("❌ Candidate is likely to be REJECTED")
        st.write(f"Confidence: **{confidence:.2f}%**")
        st.progress(float(probability[0]))

        st.subheader("⭐ Candidate Rating")
        st.warning("⭐ Candidate Needs Improvement")

    # ----------------------------
    # PDF Download (Common for both)
    # ----------------------------
    pdf = create_pdf(
        skills,
        experience,
        education,
        certification,
        job_role,
        salary,
        projects,
        ai_score,
        result,
        confidence
    )

    st.download_button(
         label="📄 Download Prediction Report",
         data=pdf,
         file_name="Hiring_Prediction_Report.pdf",
         mime="application/pdf"
)

# =====================================
# Dataset Analytics
# =====================================

st.markdown("---")
st.header("📊 Dataset Analytics")

fig1 = px.histogram(
    df,
    x="Experience (Years)",
    color="Education",
    title="Experience Distribution",
    template="plotly_white"
)

st.plotly_chart(fig1, use_container_width=True)

fig2 = px.pie(
    df,
    names="Education",
    title="Education Distribution",
    hole=0.45,
    template="plotly_white"
)

st.plotly_chart(fig2, use_container_width=True)

# Footer

st.markdown("---")

st.info("""
### 📌 Project Information

**Project Name:** AI Hiring Prediction System

**Machine Learning Algorithm:** Random Forest Classifier

**Framework:** Streamlit

**Language:** Python

**Developer:** Suhasini Singh
""")
st.caption("© 2026 AI Hiring Prediction System | Developed by Suhasini Singh using Python, Streamlit & Machine Learning")