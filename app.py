
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
