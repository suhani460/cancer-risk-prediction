
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# ---------------------------------
# PAGE CONFIGURATION
# ---------------------------------

st.set_page_config(
    page_title="Cancer Risk Prediction",
    page_icon="🧬",
    layout="wide"
)


# ---------------------------------
# CUSTOM STYLING
# ---------------------------------

st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}

h1 {
    color: #17365D;
}

h2, h3 {
    color: #24476B;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}

.stButton button {
    background-color: #17365D;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------
# DATASET CONFIGURATION
# ---------------------------------

DATA_URL = (
    "https://huggingface.co/datasets/"
    "tarekmasryo/cancer-risk-factors-data/"
    "resolve/main/data/cancer-risk-factors.csv"
)

FEATURES = [
    "Age",
    "Gender",
    "Smoking",
    "Alcohol_Use",
    "Obesity",
    "Family_History",
    "Diet_Red_Meat",
    "Diet_Salted_Processed",
    "Fruit_Veg_Intake",
    "Physical_Activity",
    "Air_Pollution",
    "Occupational_Hazards",
    "BRCA_Mutation",
    "H_Pylori_Infection",
    "Calcium_Intake",
    "BMI",
    "Physical_Activity_Level"
]


# ---------------------------------
# MODEL TRAINING
# ---------------------------------

@st.cache_resource
def train_model():

    df = pd.read_csv(DATA_URL)

    X = df[FEATURES].copy()
    y = df["Risk_Level"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        (
            "smote",
            SMOTE(random_state=42)
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=447,
                max_depth=10,
                min_samples_split=6,
                min_samples_leaf=1,
                max_features="sqrt",
                bootstrap=False,
                criterion="entropy",
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro"
    )

    return (
        model,
        df,
        accuracy,
        macro_f1,
        y_test,
        y_pred
    )


# ---------------------------------
# HEADER
# ---------------------------------

st.title("🧬 Cancer Risk Level Prediction")

st.markdown(
    "### Machine Learning Based Risk Classification System"
)

st.write(
    "This application demonstrates a Random Forest "
    "classification pipeline with SMOTE for class balancing."
)

st.warning(
    "Educational and research demonstration only. "
    "This model is not clinically validated and must not "
    "be used for medical diagnosis or treatment decisions."
)


# ---------------------------------
# LOAD MODEL
# ---------------------------------

try:

    with st.spinner(
        "Downloading dataset and training machine learning model..."
    ):

        (
            model,
            df,
            accuracy,
            macro_f1,
            y_test,
            y_pred
        ) = train_model()

    st.success("Dataset loaded and model trained successfully!")

except Exception as error:

    st.error(f"Application Error: {error}")

    st.info(
        "Please check your internet connection, dataset URL "
        "and installed Python libraries."
    )

    st.stop()


# ---------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------

st.sidebar.title("🧬 Navigation")

page = st.sidebar.radio(
    "Select a page",
    [
        "Dashboard",
        "Risk Prediction",
        "Model Evaluation",
        "Dataset Explorer"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Model: Random Forest Classifier\n\n"
    "Imbalance Handling: SMOTE"
)


# =================================
# PAGE 1: DASHBOARD
# =================================

if page == "Dashboard":

    st.header("📊 Project Dashboard")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Records",
        f"{len(df):,}"
    )

    col2.metric(
        "Accuracy",
        f"{accuracy:.2%}"
    )

    col3.metric(
        "Macro F1 Score",
        f"{macro_f1:.4f}"
    )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Risk Level Distribution")

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.countplot(
            data=df,
            x="Risk_Level",
            ax=ax
        )

        ax.set_xlabel("Risk Level")
        ax.set_ylabel("Number of Records")

        plt.xticks(rotation=20)

        st.pyplot(fig)
        plt.close(fig)

    with right:

        st.subheader("Age Distribution")

        fig, ax = plt.subplots(figsize=(7, 4))

        sns.histplot(
            df["Age"],
            bins=20,
            kde=True,
            ax=ax
        )

        ax.set_xlabel("Age")
        ax.set_ylabel("Frequency")

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# =================================
# PAGE 2: RISK PREDICTION
# =================================

elif page == "Risk Prediction":

    st.header("🔍 Predict Risk Category")

    st.write(
        "Enter the feature values below to generate "
        "a prediction from the trained model."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Personal Information")

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=120,
                value=45
            )

            gender = st.selectbox(
                "Gender",
                [0, 1],
                format_func=lambda x:
                    "Female (0)" if x == 0 else "Male (1)"
            )

            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=60.0,
                value=24.0
            )

            family = st.selectbox(
                "Family History",
                [0, 1]
            )

            brca = st.selectbox(
                "BRCA Mutation",
                [0, 1]
            )

            pylori = st.selectbox(
                "H. Pylori Infection",
                [0, 1]
            )

        with col2:

            st.subheader("Lifestyle and Environmental Factors")

            smoking = st.slider("Smoking", 0, 10, 3)

            alcohol = st.slider(
                "Alcohol Use",
                0, 10, 2
            )

            obesity = st.slider(
                "Obesity",
                0, 10, 3
            )

            red_meat = st.slider(
                "Red Meat Diet",
                0, 10, 3
            )

            processed = st.slider(
                "Salted/Processed Food",
                0, 10, 3
            )

            fruit = st.slider(
                "Fruit and Vegetable Intake",
                0, 10, 5
            )

            activity = st.slider(
                "Physical Activity",
                0, 10, 5
            )

            activity_level = st.slider(
                "Physical Activity Level",
                0, 10, 5
            )

            pollution = st.slider(
                "Air Pollution",
                0, 10, 3
            )

            hazards = st.slider(
                "Occupational Hazards",
                0, 10, 2
            )

            calcium = st.slider(
                "Calcium Intake",
                0, 10, 5
            )

        submitted = st.form_submit_button(
            "Predict Risk Level",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Smoking": smoking,
            "Alcohol_Use": alcohol,
            "Obesity": obesity,
            "Family_History": family,
            "Diet_Red_Meat": red_meat,
            "Diet_Salted_Processed": processed,
            "Fruit_Veg_Intake": fruit,
            "Physical_Activity": activity,
            "Air_Pollution": pollution,
            "Occupational_Hazards": hazards,
            "BRCA_Mutation": brca,
            "H_Pylori_Infection": pylori,
            "Calcium_Intake": calcium,
            "BMI": bmi,
            "Physical_Activity_Level": activity_level
        }])

        prediction = model.predict(input_data)[0]

        st.divider()

        st.subheader("Prediction Result")

        st.success(
            f"Predicted Dataset Risk Category: {prediction}"
        )

        st.caption(
            "This is a model classification, not an "
            "individual medical risk assessment."
        )


# =================================
# PAGE 3: MODEL EVALUATION
# =================================

elif page == "Model Evaluation":

    st.header("📈 Model Evaluation")

    col1, col2 = st.columns(2)

    col1.metric(
        "Accuracy",
        f"{accuracy:.2%}"
    )

    col2.metric(
        "Macro F1 Score",
        f"{macro_f1:.4f}"
    )

    st.divider()

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df,
        use_container_width=True
    )

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=model.classes_
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    st.pyplot(fig)
    plt.close(fig)


# =================================
# PAGE 4: DATASET EXPLORER
# =================================

elif page == "Dataset Explorer":

    st.header("📂 Dataset Explorer")

    col1, col2 = st.columns(2)

    col1.metric(
        "Rows",
        df.shape[0]
    )

    col2.metric(
        "Columns",
        df.shape[1]
    )

    st.subheader("Dataset Information")

    st.write("Column Names")

    st.write(df.columns.tolist())

    st.subheader("Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.subheader("Missing Values")

    missing = df.isnull().sum()

    st.dataframe(
        missing.rename("Missing Values"),
        use_container_width=True
    )

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Dataset CSV",
        data=csv,
        file_name="cancer-risk-factors.csv",
        mime="text/csv"
    )


# ---------------------------------
# FOOTER
# ---------------------------------

st.divider()

st.caption(
    "Cancer Risk Level Prediction | Machine Learning Project"
)
