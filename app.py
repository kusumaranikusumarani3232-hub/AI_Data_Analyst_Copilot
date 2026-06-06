import streamlit as st
import pandas as pd
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Data Analyst Copilot",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI Data Analyst Copilot")
st.write("Upload a CSV file and get AI-powered insights using Gemini.")

# -----------------------------
# GEMINI API KEY
# -----------------------------
api_key = st.sidebar.text_input(
    "Enter Gemini API Key",
    type="password"
)

if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded Successfully!")

    # -----------------------------
    # DATA PREVIEW
    # -----------------------------
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # BASIC INFO
    # -----------------------------
    st.subheader("Dataset Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    st.write("### Column Names")
    st.write(df.columns.tolist())

    st.write("### Data Types")
    st.dataframe(df.dtypes.astype(str))

    # -----------------------------
    # MISSING VALUES
    # -----------------------------
    st.subheader("Missing Values")

    missing = df.isnull().sum()
    st.dataframe(missing[missing > 0])

    # -----------------------------
    # STATISTICS
    # -----------------------------
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

    # -----------------------------
    # HISTOGRAMS
    # -----------------------------
    st.subheader("Histograms")

    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        df[numeric_cols].hist(ax=ax)
        st.pyplot(fig)

    # -----------------------------
    # HEATMAP
    # -----------------------------
    st.subheader("Correlation Heatmap")

    if len(numeric_cols) > 1:

        corr = df[numeric_cols].corr()

        fig2, ax2 = plt.subplots(figsize=(10, 6))

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm",
            ax=ax2
        )

        st.pyplot(fig2)

    # -----------------------------
    # AI INSIGHTS
    # -----------------------------
    if api_key:

        st.subheader("AI Business Insights")

        if st.button("Generate Insights"):

            prompt = f"""
            Analyze this dataset.

            Dataset Shape:
            {df.shape}

            Columns:
            {list(df.columns)}

            Missing Values:
            {df.isnull().sum().to_string()}

            Sample Data:
            {df.head(30).to_string()}

            Provide:
            1. Key Insights
            2. Trends
            3. Business Recommendations
            4. Data Quality Issues
            """

            with st.spinner("Generating insights..."):

                response = model.generate_content(prompt)

                st.write(response.text)

        # -----------------------------
        # ASK QUESTIONS
        # -----------------------------
        st.subheader("Ask Questions About Your Data")

        question = st.text_input(
            "Enter your question"
        )

        if st.button("Ask Gemini"):

            prompt = f"""
            You are a professional Data Analyst.

            Dataset Shape:
            {df.shape}

            Columns:
            {list(df.columns)}

            Sample Data:
            {df.head(30).to_string()}

            Question:
            {question}

            Give a detailed answer.
            """

            with st.spinner("Analyzing..."):

                response = model.generate_content(prompt)

                st.write(response.text)

else:
    st.info("Upload a CSV file to begin.")
