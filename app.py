import streamlit as st
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page Configuration
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Screening System")
st.write("Upload a resume and compare it against a job description.")

# Upload Resume
uploaded_file = st.file_uploader(
    "Upload Resume (PDF)",
    type=["pdf"]
)

# Job Description
job_description = st.text_area(
    "Enter Job Description",
    height=150
)

if uploaded_file:

    resume_text = ""

    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()

                if page_text:
                    resume_text += page_text + "\n"

        st.success("✅ Resume Uploaded Successfully!")

        with st.expander("📄 View Extracted Resume Text"):
            st.text_area(
                "Resume Content",
                resume_text[:5000],
                height=300
            )

        if st.button("🔍 Analyze Resume"):

            if job_description.strip() == "":
                st.warning("Please enter a Job Description.")
                st.stop()

            # AI Matching
            documents = [resume_text, job_description]

            tfidf = TfidfVectorizer()

            matrix = tfidf.fit_transform(documents)

            similarity = cosine_similarity(
                matrix[0:1],
                matrix[1:2]
            )

            match_score = similarity[0][0] * 100

            st.subheader("🎯 Resume Match Score")

            st.progress(min(int(match_score), 100))

            st.success(f"{match_score:.2f}% Match")

            # Skill Database
            skills = [
                "Python",
                "SQL",
                "Machine Learning",
                "Deep Learning",
                "Data Analysis",
                "Power BI",
                "Excel",
                "HTML",
                "CSS",
                "Java",
                "AWS",
                "Docker",
                "Git",
                "GitHub",
                "REST API",
                "Pandas",
                "NumPy",
                "TensorFlow",
                "PyTorch",
                "JavaScript",
                "React",
                "Flask",
                "Django",
                "OpenCV",
                "NLP"
            ]

            matched = []
            missing = []

            for skill in skills:

                resume_has_skill = skill.lower() in resume_text.lower()
                jd_has_skill = skill.lower() in job_description.lower()

                if resume_has_skill and jd_has_skill:
                    matched.append(skill)

                elif jd_has_skill:
                    missing.append(skill)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("✅ Matched Skills")

                if matched:
                    for skill in matched:
                        st.write("✔", skill)
                else:
                    st.write("No matching skills found.")

            with col2:
                st.subheader("❌ Missing Skills")

                if missing:
                    for skill in missing:
                        st.write("✖", skill)
                else:
                    st.write("No missing skills.")

            # ATS Score
            ats_score = 0

            ats_score += min(len(matched) * 10, 50)

            if "projects" in resume_text.lower():
                ats_score += 15

            if "certificate" in resume_text.lower():
                ats_score += 15

            if "github" in resume_text.lower():
                ats_score += 10

            if "linkedin" in resume_text.lower():
                ats_score += 10

            ats_score = min(ats_score, 100)

            st.subheader("⭐ ATS Score")

            st.progress(ats_score / 100)

            st.success(f"{ats_score}/100")

            # Resume Strengths
            st.subheader("💪 Resume Strengths")

            strengths = []

            if "python" in resume_text.lower():
                strengths.append("Strong Python Background")

            if "machine learning" in resume_text.lower():
                strengths.append("Machine Learning Knowledge")

            if "projects" in resume_text.lower():
                strengths.append("Contains Projects")

            if "certificate" in resume_text.lower():
                strengths.append("Contains Certifications")

            if "github" in resume_text.lower():
                strengths.append("GitHub Profile Included")

            if "linkedin" in resume_text.lower():
                strengths.append("LinkedIn Profile Included")

            for item in strengths:
                st.write("✅", item)

            # Suggestions
            st.subheader("📌 Suggestions for Improvement")

            suggestions = []

            if "aws" not in resume_text.lower():
                suggestions.append("Learn AWS and add cloud projects.")

            if "docker" not in resume_text.lower():
                suggestions.append("Learn Docker and mention containerization skills.")

            if "git" not in resume_text.lower():
                suggestions.append("Add Git/GitHub experience.")

            if "power bi" not in resume_text.lower():
                suggestions.append("Consider learning Power BI for analytics roles.")

            if match_score < 50:
                suggestions.append("Customize your resume for each job description.")

            if suggestions:
                for s in suggestions:
                    st.write("•", s)
            else:
                st.write("No major improvements needed.")

            # Interview Questions
            st.subheader("🎤 Possible Interview Questions")

            questions = []

            if "python" in resume_text.lower():
                questions.append("What are Python decorators?")

            if "sql" in resume_text.lower():
                questions.append("Explain INNER JOIN vs LEFT JOIN.")

            if "machine learning" in resume_text.lower():
                questions.append("What is overfitting in Machine Learning?")

            if "deep learning" in resume_text.lower():
                questions.append("Difference between CNN and ANN?")

            if "rest api" in resume_text.lower():
                questions.append("What is the difference between GET and POST requests?")

            if "data analysis" in resume_text.lower():
                questions.append("How do you handle missing data in a dataset?")

            for q in questions:
                st.write("•", q)

    except Exception as e:
        st.error(f"Error reading PDF: {e}")