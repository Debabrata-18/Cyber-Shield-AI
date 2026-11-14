
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from datetime import datetime

# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Cyber Shield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------
# LOAD AI MODEL
# ----------------------------------

try:
    model = joblib.load("cyber_model.pkl")
    model_status = True
except Exception as e:
    model = None
    model_status = False
    model_error = str(e)

# ----------------------------------
# CUSTOM DESIGN
# ----------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
    color: #e5eaf3;
}
[data-testid="stSidebar"] {
    background-color: #111c30;
}
h1, h2, h3 {
    color: #e5eaf3 !important;
}
p, label {
    color: #b9c5d8;
}
div[data-testid="stMetric"] {
    background: #17243a;
    border: 1px solid #263c59;
    padding: 18px;
    border-radius: 12px;
}
div[data-testid="stMetricValue"] {
    color: #40c4ff;
}
.stButton > button {
    background-color: #0878d1;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
}
div[data-testid="stTextArea"] textarea {
    background-color: #17243a;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# LOAD DATASET
# ----------------------------------

@st.cache_data
def load_data():
    try:
        data = pd.read_csv("cleaned_dataset.csv")
    except FileNotFoundError:
        try:
            data = pd.read_csv("dataset.csv")
        except FileNotFoundError:
            return pd.DataFrame(columns=["message", "label"])

    if not {"message", "label"}.issubset(data.columns):
        return pd.DataFrame(columns=["message", "label"])

    data = data.dropna(subset=["message", "label"])
    data["message"] = data["message"].astype(str).str.strip()
    data["label"] = data["label"].astype(str).str.strip()

    return data

df = load_data()

# ----------------------------------
# SESSION HISTORY
# ----------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------------
# SIDEBAR
# ----------------------------------

st.sidebar.title("🛡️ CYBER SHIELD AI")
st.sidebar.caption("Smart Cybercrime Detection System")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Message Scanner",
        "Dataset Analytics",
        "Detection History",
        "About Project"
    ]
)

st.sidebar.divider()
st.sidebar.caption("PROJECT STATUS")
st.sidebar.success("Dashboard Online")

if model_status:
    st.sidebar.success("AI Model: Loaded")
else:
    st.sidebar.error("AI Model: Not available")

# ----------------------------------
# DASHBOARD
# ----------------------------------

if page == "Dashboard":

    st.title("Security Intelligence Dashboard")
    st.caption("Monitor cybersecurity data and analyse potential threats.")
    st.divider()

    total = len(df)
    labels = df["label"].str.lower()

    safe = int(labels.eq("safe").sum())
    suspicious = int(labels.eq("suspicious").sum())
    malicious = int(labels.isin(["malicious", "phishing"]).sum())

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", total)
    col2.metric("Safe Records", safe)
    col3.metric("Suspicious", suspicious)
    col4.metric("Malicious", malicious)

    left, right = st.columns(2)

    with left:
        st.subheader("Threat Distribution")

        if total > 0:
            chart_data = df["label"].value_counts().reset_index()
            chart_data.columns = ["Category", "Count"]

            fig = px.pie(
                chart_data,
                names="Category",
                values="Count",
                hole=0.55
            )

            fig.update_layout(
                paper_bgcolor="#0b1220",
                plot_bgcolor="#0b1220",
                font_color="white"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No dataset available.")

    with right:
        st.subheader("Security Summary")

        summary = pd.DataFrame({
            "Category": ["Safe", "Suspicious", "Malicious"],
            "Records": [safe, suspicious, malicious]
        })

        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("Recent Dataset Records")

    if total > 0:
        st.dataframe(df.tail(5), use_container_width=True, hide_index=True)

# ----------------------------------
# MESSAGE SCANNER
# ----------------------------------

elif page == "Message Scanner":

    st.title("Message Threat Scanner")

    st.caption("Enter a message for experimental AI classification.")

    message = st.text_area(
        "Paste your message or email",
        height=160,
        placeholder="Enter a sample message here..."
    )

    if st.button("Analyse Message"):

        if not message.strip():
            st.error("Please enter a message.")

        elif model is None:
            st.error("AI model could not be loaded.")

        else:
            prediction = model.predict([message])[0]
            probabilities = model.predict_proba([message])[0]
            confidence = max(probabilities) * 100

            st.subheader("AI Analysis Result")

            if prediction == "Safe":
                st.success("Prediction: SAFE")
            elif prediction == "Suspicious":
                st.warning("Prediction: SUSPICIOUS")
            else:
                st.error("Prediction: MALICIOUS")

            st.metric("Model confidence", f"{confidence:.2f}%")

            st.session_state.history.append({
                "Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                "Message": message[:100],
                "Status": prediction
            })

            st.info(
                "Experimental educational model trained on 100 labelled sample messages. "
                "Predictions are not reliable security assessments."
            )

    st.info(
        "Never enter real passwords, OTPs, banking details "
        "or private personal information."
    )

# ----------------------------------
# DATASET ANALYTICS
# ----------------------------------

elif page == "Dataset Analytics":

    st.title("Dataset Analytics")

    if not df.empty:

        st.subheader("Dataset Preview")
        st.dataframe(df, use_container_width=True, hide_index=True)

        counts = df["label"].value_counts().reset_index()
        counts.columns = ["Category", "Records"]

        fig = px.bar(
            counts,
            x="Category",
            y="Records",
            color="Category",
            title="Training Dataset Distribution"
        )

        fig.update_layout(
            paper_bgcolor="#0b1220",
            plot_bgcolor="#0b1220",
            font_color="white"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download Dataset",
            df.to_csv(index=False),
            file_name="cyber_dataset.csv",
            mime="text/csv"
        )

    else:
        st.warning("Dataset not found.")

# ----------------------------------
# DETECTION HISTORY
# ----------------------------------

elif page == "Detection History":

    st.title("Detection History")
    st.caption("Monitor previous AI message classifications.")

    if st.session_state.history:

        history_df = pd.DataFrame(st.session_state.history)

        # Statistics
        status = history_df["Status"].astype(str).str.strip().str.title()

        total_scans = len(history_df)
        safe_count = int(status.eq("Safe").sum())
        suspicious_count = int(status.eq("Suspicious").sum())
        malicious_count = int(status.eq("Malicious").sum())

        st.subheader("Detection Statistics")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Scans", total_scans)
        col2.metric("Safe", safe_count)
        col3.metric("Suspicious", suspicious_count)
        col4.metric("Malicious", malicious_count)

        st.divider()

        # Charts
        st.subheader("Interactive Threat Analysis")

        chart_data = pd.DataFrame({
            "Category": ["Safe", "Suspicious", "Malicious"],
            "Count": [safe_count, suspicious_count, malicious_count]
        })

        left, right = st.columns(2)

        with left:
            st.write("Threat Distribution")

            nonzero_data = chart_data[chart_data["Count"] > 0]

            if not nonzero_data.empty:
                pie_fig = px.pie(
                    nonzero_data,
                    names="Category",
                    values="Count",
                    hole=0.5,
                    title="Classification Distribution"
                )

                pie_fig.update_layout(
                    paper_bgcolor="#0b1220",
                    plot_bgcolor="#0b1220",
                    font_color="white"
                )

                st.plotly_chart(pie_fig, use_container_width=True)

        with right:
            st.write("Category Comparison")

            bar_fig = px.bar(
                chart_data,
                x="Category",
                y="Count",
                color="Category",
                text="Count",
                title="Number of Detections"
            )

            bar_fig.update_layout(
                paper_bgcolor="#0b1220",
                plot_bgcolor="#0b1220",
                font_color="white",
                showlegend=False
            )

            st.plotly_chart(bar_fig, use_container_width=True)

        st.divider()

        # Search
        st.subheader("Detection Records")

        search_text = st.text_input(
            "Search detection history",
            placeholder="Search by message or category..."
        )

        filtered_history = history_df.copy()

        if search_text.strip():
            mask = (
                filtered_history["Message"].str.contains(
                    search_text, case=False, na=False
                )
                |
                filtered_history["Status"].str.contains(
                    search_text, case=False, na=False
                )
            )

            filtered_history = filtered_history[mask]

        st.write(f"Showing {len(filtered_history)} of {total_scans} records")

        st.dataframe(
            filtered_history,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Detection History CSV",
            history_df.to_csv(index=False),
            file_name="cyber_detection_history.csv",
            mime="text/csv"
        )

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()

    else:
        st.info(
            "No detection records yet. Visit Message Scanner "
            "and analyse a sample message first."
        )

# ----------------------------------
# ABOUT PROJECT
# ----------------------------------

elif page == "About Project":

    st.title("About Cyber Shield AI")

    st.write("""
    Cyber Shield AI is an educational Artificial Intelligence
    project developed to explore cybersecurity threat detection.

    The project uses Machine Learning to classify sample messages
    as Safe, Suspicious or Malicious.
    """)

    st.subheader("Technology Used")

    st.markdown("""
    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - Plotly
    - Machine Learning
    """)

    st.subheader("Project Objective")

    st.write(
        "To develop an AI-assisted educational system "
        "that helps users recognise potential cyber threats."
    )

# ----------------------------------
# FOOTER
# ----------------------------------

st.divider()

st.caption(
    "CYBER SHIELD AI | CBSE CLASS 12 AI PROJECT | 2026-27"
)
