
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from urllib.request import Request, urlopen
from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="Ghost in the Dataset",
    page_icon="👻",
    layout="wide"
)

st.title("👻 Ghost in the Dataset")
st.subheader("Your AI-Powered Data Detective")

st.markdown(
    "Discover hidden patterns and mysterious anomalies in your data."
)

st.sidebar.title("🔎 Investigation Panel")

uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("Dataset successfully loaded!")

    st.header("📂 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] > 0 and len(numeric_df) >= 5:

        st.subheader("👻 Ghost Detection")

        clean_data = numeric_df.fillna(numeric_df.median())

        model = IsolationForest(
            contamination=0.05,
            random_state=42
        )

        predictions = model.fit_predict(clean_data)

        result = df.copy()
        result["Ghost Status"] = np.where(
            predictions == -1,
            "👻 Anomaly",
            "Normal"
        )

        anomalies = result[
            result["Ghost Status"] == "👻 Anomaly"
        ]

        st.metric("Mysterious Records Found", len(anomalies))

        st.dataframe(result, use_container_width=True)

        st.subheader("📊 Data Visualization")

        selected_column = st.selectbox(
            "Choose a numerical column",
            numeric_df.columns.tolist()
        )

        fig, ax = plt.subplots()

        ax.hist(
            numeric_df[selected_column].dropna(),
            bins=20
        )

        ax.set_title(f"Distribution of {selected_column}")
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Frequency")

        st.pyplot(fig)

        st.subheader("🕵️ Investigation Report")

        st.write(
            f"Investigated {len(df)} records and "
            f"detected {len(anomalies)} unusual records."
        )

        # Gemma AI Integration
        st.subheader("🤖 Ask Ghost AI - Powered by Gemma")

        st.write(
            "Let Gemma investigate your dataset "
            "and explain the unusual patterns."
        )

        if st.button("👻 Investigate with Gemma AI"):

            prompt = f"""
            You are Ghost, a careful and helpful data analyst.

            Analyze these dataset investigation results.

            Total records: {len(df)}
            Total columns: {len(df.columns)}
            Missing values: {int(df.isnull().sum().sum())}
            Anomalies detected: {len(anomalies)}

            Numerical data summary:
            {numeric_df.describe().round(2).to_string()}

            Sample unusual records:
            {anomalies.head(5).drop(
                columns=["Ghost Status"]
            ).to_string(index=False)}

            Explain:
            1. What unusual patterns were detected?
            2. What could explain these unusual values?
            3. What should a data analyst investigate next?

            Do not assume anomalies are errors or wrongdoing.
            Do not claim causation without evidence.
            Keep the explanation clear and concise.
            """

            try:
                with st.spinner(
                    "👻 Gemma is investigating your dataset..."
                ):
                    payload = json.dumps({
                        "model": "gemma2:2b",
                        "prompt": prompt,
                        "stream": False
                    }).encode("utf-8")

                    request = Request(
                        "http://localhost:11434/api/generate",
                        data=payload,
                        headers={
                            "Content-Type": "application/json"
                        }
                    )

                    with urlopen(request, timeout=180) as response:
                        result_json = json.loads(
                            response.read().decode("utf-8")
                        )

                    st.success("Ghost investigation completed!")
                    st.write(result_json.get("response", "No response received."))

            except Exception as error:
                st.error(f"Gemma connection error: {error}")
                st.info(
                    "Please check that Ollama is running "
                    "and Gemma 2B is installed."
                )

        csv = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Investigation Report",
            csv,
            "ghost_investigation.csv",
            "text/csv"
        )

    else:
        st.warning(
            "Please upload a dataset with at least 5 rows "
            "and one numerical column."
        )

else:
    st.info("Upload a CSV file to begin your investigation.")
    st.write("👻 The ghosts are waiting inside your dataset...")
