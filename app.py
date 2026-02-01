import os
import io
import re

import streamlit as st
import pandas as pd
import numpy as np

import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from openai import OpenAI

# ------------------------------------------------
# Basic page config
# ------------------------------------------------
st.set_page_config(
    page_title="RetentionGPT – Telco Churn Assistant",
    page_icon="📉",
    layout="wide"
)

st.title("RetentionGPT – Telco Churn & Retention Assistant")
st.caption("Upload a telco dataset, score customers, and ask an AI assistant for retention strategy.")

DATA_DIR = "data"
TRAIN_PATH = os.path.join(DATA_DIR, "final_data.csv")

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "telco_churn_model.joblib")

TARGET_DEFAULT = "churn_flag"

# ------------------------------------------------
# Column helpers / cleaning
# ------------------------------------------------
def normalize_colname(c: str) -> str:
    c = c.strip()
    c = re.sub(r"\s+", "_", c)
    c = re.sub(r"[^\w_]", "", c)
    return c.lower()


def normalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_colname(c) for c in df.columns]
    return df


def standardize_missing_tokens(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    missing_tokens = {"", "na", "n/a", "nan", "null", "none", "?", "unknown"}
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].replace({t: np.nan for t in missing_tokens})
    return df


def yes_no_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mapping = {
        "yes": 1, "y": 1, "true": 1, "t": 1,
        "no": 0, "n": 0, "false": 0, "f": 0
    }
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].astype(str).str.lower().str.strip()
            uniq = set(s.dropna().unique().tolist())
            if uniq and uniq.issubset(set(mapping.keys())):
                df[c] = s.map(mapping).astype("float")
    return df


def money_like_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].astype(str)
            # detect if column looks like money/number with $ or commas
            sample = s.dropna().head(50)
            if len(sample) == 0:
                continue
            looks_money = sample.str.contains(r"[\$,\d]", regex=True).mean() > 0.6
            if looks_money:
                cleaned = (
                    s.str.replace(r"[\$,]", "", regex=True)
                     .str.replace(r"\s+", "", regex=True)
                )
                # coercing to numeric can create many NaNs; only keep if it converts decently
                numeric = pd.to_numeric(cleaned, errors="coerce")
                if numeric.notna().mean() > 0.5:
                    df[c] = numeric
    return df


def drop_high_cardinality_strings(df: pd.DataFrame, threshold: int = 200) -> pd.DataFrame:
    df = df.copy()
    drop_cols = []
    for c in df.columns:
        if df[c].dtype == object:
            nunique = df[c].nunique(dropna=True)
            if nunique > threshold:
                drop_cols.append(c)
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_df_columns(df)
    df = standardize_missing_tokens(df)
    df = yes_no_to_binary(df)
    df = money_like_to_numeric(df)
    df = drop_high_cardinality_strings(df)
    return df


def file_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.isfile(path)


def load_training_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = basic_clean(df)
    return df


def split_features_target(df: pd.DataFrame, target: str = TARGET_DEFAULT):
    df = df.copy()
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in training data.")
    y = df[target].astype(int)

    # drop noisy / non-feature columns if present
    drop_cols = [target]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    return X, y, num_cols, cat_cols


def train_churn_model(df: pd.DataFrame, target: str = TARGET_DEFAULT):
    """
    Train churn model on the training dataset and return:
    - model pipeline
    - feature_cols (exact training feature list, in order)
    """
    X, y, num_cols, cat_cols = split_features_target(df, target=target)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )

    clf = LogisticRegression(max_iter=500)

    model = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", clf),
    ])

    model.fit(X, y)

    return model, X.columns.tolist()


def decide_mode(upload_df: pd.DataFrame, training_feature_cols: list[str], target: str = TARGET_DEFAULT):
    """
    Returns one of:
    - "pretrained"        -> score with our pre-trained model
    - "train_on_upload"   -> train a model on the upload (needs target)
    - "block"             -> too different + no target
    """
    cols = set(upload_df.columns)
    train_cols = set(training_feature_cols)

    has_target = target in cols
    overlap = len(cols.intersection(train_cols)) / max(1, len(train_cols))

    if has_target:
        return "train_on_upload"
    if overlap >= 0.30:
        return "pretrained"
    return "block"


def align_schema_for_model(df: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    """
    Build a dataframe with exactly the columns expected by the model.
    Extra columns are ignored. Missing columns are created as NaN.
    """
    df = df.copy()
    aligned = pd.DataFrame(index=df.index)
    for c in expected_cols:
        if c in df.columns:
            aligned[c] = df[c]
        else:
            aligned[c] = np.nan
    return aligned


def score_with_model(model, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    X = align_schema_for_model(df, feature_cols)

    proba = model.predict_proba(X)[:, 1]
    df["predicted_churn_proba"] = proba
    return df


def assign_risk_bands(df: pd.DataFrame, proba_col: str = "predicted_churn_proba") -> pd.DataFrame:
    df = df.copy()
    try:
        df["risk_band"] = pd.qcut(
            df[proba_col],
            q=4,
            labels=["Low", "Medium", "High", "Very High"]
        )
    except Exception:
        df["risk_band"] = pd.cut(
            df[proba_col],
            bins=[-0.001, 0.25, 0.50, 0.75, 1.001],
            labels=["Low", "Medium", "High", "Very High"]
        )
    return df


def compute_summary(df: pd.DataFrame) -> dict:
    proba = df["predicted_churn_proba"]
    return {
        "total_customers": int(len(df)),
        "avg_risk": float(proba.mean()) if len(proba) else 0.0,
        "top_10pct_threshold": float(proba.quantile(0.90)) if len(proba) else 0.0
    }


def build_context(scored_df: pd.DataFrame) -> str:
    """
    Build compact context for the LLM: distribution + top risk customers.
    """
    if scored_df is None or scored_df.empty:
        return "No data scored."

    summary = compute_summary(scored_df)
    band_counts = scored_df["risk_band"].value_counts(dropna=False).to_dict()

    top = scored_df.sort_values("predicted_churn_proba", ascending=False).head(25)
    top_rows = top.to_csv(index=False)

    context = f"""
You are a retention strategy assistant.

Summary:
- Total customers: {summary['total_customers']}
- Average churn risk: {summary['avg_risk']:.3f}
- Top 10% churn risk threshold: {summary['top_10pct_threshold']:.3f}

Risk band counts:
{band_counts}

Top risky customers (sample):
{top_rows}

Now answer the user's question with concrete retention actions.
"""
    return context.strip()


def init_openai_client():
    try:
        if "OPENAI_API_KEY" not in st.secrets:
            return None
        key = st.secrets["OPENAI_API_KEY"]
        if not key:
            return None
        return OpenAI(api_key=key)
    except Exception as e:
        st.warning(f"OpenAI client init failed; falling back to rule-based assistant. Details: {e}")
        return None


# ------------------------------------------------
# Load base training data and load the bundled model
# ------------------------------------------------
if not file_exists(TRAIN_PATH):
    st.error(f"Training file not found at `{TRAIN_PATH}`. Please add final_data.csv to the data/ folder.")
    st.stop()

base_df = load_training_data(TRAIN_PATH)

if not file_exists(MODEL_PATH):
    st.error(f"Model file not found at `{MODEL_PATH}`. Please add telco_churn_model.joblib to the models/ folder.")
    st.stop()

model = joblib.load(MODEL_PATH)

# Use model-native feature list when available; otherwise fall back to training schema (minus target)
feature_cols = getattr(model, "feature_names_in_", None)
if feature_cols is None:
    feature_cols = [c for c in base_df.columns if c != TARGET_DEFAULT]
else:
    feature_cols = list(feature_cols)

# ------------------------------------------------
# Sidebar – upload new data
# ------------------------------------------------
st.sidebar.header("Data settings")
st.sidebar.write("Upload a telco CSV to run churn scoring + AI analysis.")

uploaded_file = st.sidebar.file_uploader("Upload new customer data (CSV)", type=["csv"])

# ---- Clear chat when dataset changes ----
if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_file_key" not in st.session_state:
    st.session_state.active_file_key = None

new_file_key = None
if uploaded_file is not None:
    # unique enough for most cases (name + size)
    new_file_key = (uploaded_file.name, uploaded_file.size)

# If user uploads a different file OR removes the file, reset chat
if new_file_key != st.session_state.active_file_key:
    st.session_state.messages = []
    st.session_state.active_file_key = new_file_key

# ------------------------------------------------
# Main: load uploaded data and run
# ------------------------------------------------
if uploaded_file is None:
    st.info("Upload a CSV from the sidebar to begin.")
    st.stop()

try:
    raw = uploaded_file.read()
    uploaded_df = pd.read_csv(io.BytesIO(raw))
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

uploaded_df = basic_clean(uploaded_df)

mode = decide_mode(uploaded_df, training_feature_cols=feature_cols, target=TARGET_DEFAULT)

if mode == "block":
    st.error(
        "This dataset looks too different from the training schema and does not include a churn target. "
        "Please upload a telco dataset with similar columns, or include the churn_flag column to train-on-upload."
    )
    st.stop()

# Score
if mode == "train_on_upload":
    # train model on upload (requires target), then score the same data
    try:
        X, y, num_cols, cat_cols = split_features_target(uploaded_df, target=TARGET_DEFAULT)

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_cols),
                ("cat", categorical_transformer, cat_cols),
            ]
        )

        clf = LogisticRegression(max_iter=500)
        temp_model = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])
        temp_model.fit(X, y)

        scored_df = uploaded_df.copy()
        scored_df = scored_df.drop(columns=[TARGET_DEFAULT], errors="ignore")
        scored_df = score_with_model(temp_model, scored_df, feature_cols=list(scored_df.columns))
        st.sidebar.success("Mode: Train-on-upload (trained on your data).")
    except Exception as e:
        st.error(f"Train-on-upload failed: {e}")
        st.stop()
else:
    scored_df = score_with_model(model, uploaded_df, feature_cols=feature_cols)
    st.sidebar.success("Mode: Pre-trained model (bundled).")

scored_df = assign_risk_bands(scored_df)

summary = compute_summary(scored_df)

# ------------------------------------------------
# Layout: tabs
# ------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Scoring Results", "💬 Retention Assistant", "ℹ️ How it works"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total customers", summary["total_customers"])
    c2.metric("Average churn risk", f"{summary['avg_risk']:.3f}")
    c3.metric("Top 10% risk threshold", f"{summary['top_10pct_threshold']:.3f}")

    st.subheader("Risk band distribution")
    band_summary = (
        scored_df["risk_band"]
        .value_counts()
        .rename_axis("risk_band")
        .reset_index(name="count")
    )
    st.dataframe(band_summary, use_container_width=True)

    st.subheader("Top 100 highest-risk customers")
    top100 = scored_df.sort_values("predicted_churn_proba", ascending=False).head(100)
    st.dataframe(top100, use_container_width=True)

    st.subheader("Download scored results")
    csv_bytes = scored_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download scored CSV",
        data=csv_bytes,
        file_name="scored_churn_results.csv",
        mime="text/csv"
    )

with tab2:
    st.write("Ask questions like:")
    st.markdown(
        "- What are the top drivers of churn risk?\n"
        "- What retention offers should we target to Very High risk customers?\n"
        "- Give me a prioritized action plan for the top 10% risky customers."
    )

    client = init_openai_client()

    if client is None:
        st.warning("No OpenAI API key found in Streamlit secrets. Using a simple rule-based assistant.")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Ask a retention question...")
    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        context = build_context(scored_df)

        if client is None:
            # Basic fallback response
            reply = (
                "Here are practical next steps:\n\n"
                "1) Focus on the Very High band first: offer contract upgrades, targeted discounts, and service outreach.\n"
                "2) For High band: reinforce value (bundles, loyalty perks) and reduce friction (billing, support).\n"
                "3) Identify patterns: check if specific contract types, payment methods, or high monthly charges dominate the top-risk group.\n\n"
                "If you share the columns present (contract, monthly charges, tenure), I can tailor the actions to your dataset schema."
            )
        else:
            try:
                messages = [
                    {"role": "system", "content": "You are a helpful retention strategy assistant."},
                    {"role": "system", "content": context},
                    {"role": "user", "content": user_prompt},
                ]
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.3,
                )
                reply = resp.choices[0].message.content
            except Exception as e:
                reply = f"LLM call failed, using fallback. Error: {e}"

        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

with tab3:
    st.subheader("What this app does")
    st.markdown(
        """
- Loads a pre-trained churn model and training schema from the app's folders.
- You upload a telco customer CSV.
- The app cleans and aligns your schema (fills missing columns with nulls, ignores extras).
- It predicts churn probability per row and assigns a risk band:
  - Low / Medium / High / Very High
- It shows summary metrics, top risky customers, and lets you download scored results.
- The assistant uses a compact summary of scored results to recommend retention actions.
"""
    )

    st.subheader("Expected folders")
    st.code(
        """project/
  App.py
  data/
    final_data.csv
  models/
    telco_churn_model.joblib
"""
    )
