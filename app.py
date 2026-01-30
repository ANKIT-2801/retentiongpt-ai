import os
import io
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from openai import OpenAI

# ------------------------------------------------
# Page config
# ------------------------------------------------
st.set_page_config(
    page_title="RetentionGPT – Telco Churn Assistant",
    page_icon="📉",
    layout="wide"
)

st.title("RetentionGPT – Telco Churn & Retention Assistant")
st.caption("Upload a telco dataset, score churn risk, and ask an AI assistant for retention strategy.")

DATA_DIR = "data"
TRAIN_PATH = os.path.join(DATA_DIR, "final_data.csv")

# ------------------------------------------------
# Helpers
# ------------------------------------------------
def file_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


@st.cache_data(show_spinner=False)
def load_training_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def split_features_target(df: pd.DataFrame):
    if "churn_flag" not in df.columns:
        raise ValueError("Training data must contain 'churn_flag'.")

    y = df["churn_flag"].astype(int)

    drop_cols = [
        "churn_flag",
        "predicted_churn_proba",
        "risk_band",
        "customer_id",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    return X, y, num_cols, cat_cols


@st.cache_resource(show_spinner=True)
def train_churn_model(df: pd.DataFrame):
    X, y, num_cols, cat_cols = split_features_target(df)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=500))
    ])

    model.fit(X, y)

    return model, X.columns.tolist()


def score_dataset(df: pd.DataFrame, model, feature_cols):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    X = pd.DataFrame(index=df.index)
    for col in feature_cols:
        X[col] = df[col] if col in df.columns else np.nan

    proba = model.predict_proba(X)[:, 1]
    df["predicted_churn_proba"] = proba

    df["risk_band"] = pd.qcut(
        df["predicted_churn_proba"],
        q=4,
        labels=["Low risk", "Medium risk", "High risk", "Very high risk"]
    )

    return df


def build_context(scored_df: pd.DataFrame):
    ctx = {}
    p = scored_df["predicted_churn_proba"]

    ctx["n_customers"] = len(scored_df)
    ctx["avg_risk"] = float(p.mean())
    ctx["p90_risk"] = float(np.percentile(p, 90))

    ctx["segment_summary"] = (
        scored_df.groupby("risk_band")["predicted_churn_proba"]
        .agg(customers="count", avg_churn_risk="mean")
        .reset_index()
        .sort_values("avg_churn_risk", ascending=False)
    )

    return ctx


# ------------------------------------------------
# LLM
# ------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_llm_client():
    api_key = st.secrets.get("OPENROUTER_API_KEY")
    if not api_key:
        return None

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )


def ask_llm(question: str, ctx: dict):
    client = get_llm_client()
    if client is None:
        return None

    context_text = f"""
Customers analysed: {ctx['n_customers']}
Average churn risk: {ctx['avg_risk']:.3f}
Top 10% threshold: {ctx['p90_risk']:.3f}

Risk bands:
{ctx['segment_summary'].to_string(index=False)}
"""

    completion = client.chat.completions.create(
        model="openrouter/auto",
        messages=[
            {
                "role": "system",
                "content": "You are a senior telecom retention analyst. Be concise and business-focused."
            },
            {
                "role": "user",
                "content": f"{context_text}\n\nQuestion: {question}"
            }
        ],
        temperature=0.4,
        max_tokens=400
    )

    return completion.choices[0].message.content.strip()


# ------------------------------------------------
# Train model (hidden from users)
# ------------------------------------------------
if not file_exists(TRAIN_PATH):
    st.error("Training file missing: data/final_data.csv")
    st.stop()

train_df = load_training_data(TRAIN_PATH)
model, feature_cols = train_churn_model(train_df)

# ------------------------------------------------
# Sidebar upload
# ------------------------------------------------
st.sidebar.header("Data settings")
st.sidebar.write("Upload a telco CSV to run churn analysis.")

uploaded_file = st.sidebar.file_uploader("Upload customer data (CSV)", type=["csv"])

scored_df = None
ctx = {}

if uploaded_file is not None:
    raw = uploaded_file.read()
    user_df = pd.read_csv(io.BytesIO(raw))
    scored_df = score_dataset(user_df, model, feature_cols)
    ctx = build_context(scored_df)

    st.sidebar.success("Dataset uploaded and scored.")
    st.session_state.messages = []

else:
    st.sidebar.info("No dataset uploaded yet.")

# ------------------------------------------------
# Tabs
# ------------------------------------------------
tab_chat, tab_data, tab_how = st.tabs(["Chat assistant", "Data preview", "How it works"])

with tab_chat:
    if scored_df is None:
        st.info("Upload a dataset to enable the assistant.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Ask about churn, segments, or retention strategy")
    if q:
        st.session_state.messages.append({"role": "user", "content": q})

        reply = ask_llm(q, ctx) or "LLM unavailable."
        with st.chat_message("assistant"):
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})


with tab_data:
    if scored_df is None:
        st.info("Upload a dataset to see results.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{ctx['n_customers']:,}")
    c2.metric("Avg churn risk", f"{ctx['avg_risk']*100:.1f}%")
    c3.metric("Top 10% threshold", f"{ctx['p90_risk']*100:.1f}%")

    st.dataframe(scored_df.head(25), use_container_width=True)
    st.dataframe(ctx["segment_summary"], use_container_width=True)


with tab_how:
    st.markdown("""
**How it works**

1. A churn model is trained once on internal telco data.
2. Users upload their own dataset.
3. The model scores churn probability and assigns risk bands.
4. An AI assistant answers questions using only the uploaded data.

Training data is never exposed to users.
""")
