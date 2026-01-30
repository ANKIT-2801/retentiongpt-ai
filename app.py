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
# Page setup
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
# Utilities
# ------------------------------------------------
def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def split_xy(df):
    y = df["churn_flag"].astype(int)

    drop_cols = ["churn_flag", "predicted_churn_proba", "risk_band", "customer_id"]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)

    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in X.columns if c not in num]

    return X, y, num, cat


@st.cache_resource
def train_model(train_df):
    X, y, num_cols, cat_cols = split_xy(train_df)

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=500))
    ])

    model.fit(X, y)
    return model, X.columns.tolist()


def score(df, model, features):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    X = pd.DataFrame(index=df.index)
    for f in features:
        X[f] = df[f] if f in df.columns else np.nan

    df["predicted_churn_proba"] = model.predict_proba(X)[:, 1]
    df["risk_band"] = pd.qcut(
        df["predicted_churn_proba"],
        4,
        labels=["Low risk", "Medium risk", "High risk", "Very high risk"]
    )
    return df


def build_summary(df):
    p = df["predicted_churn_proba"]

    return {
        "customers": len(df),
        "avg_risk": float(p.mean()),
        "p90": float(np.percentile(p, 90)),
        "segments": (
            df.groupby("risk_band")["predicted_churn_proba"]
            .agg(customers="count", avg_risk="mean")
            .reset_index()
            .sort_values("avg_risk", ascending=False)
        )
    }


# ------------------------------------------------
# LLM
# ------------------------------------------------
@st.cache_resource
def get_llm():
    key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not key:
        return None

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key
    )


def ask_llm(question, summary):
    client = get_llm()
    if client is None:
        raise RuntimeError("LLM not connected")

    context = f"""
Customers: {summary['customers']}
Average churn risk: {summary['avg_risk']:.3f}
Top 10% threshold: {summary['p90']:.3f}

Risk segments:
{summary['segments'].to_string(index=False)}
"""

    res = client.chat.completions.create(
        model="google/gemini-1.5-flash",
        messages=[
            {"role": "system", "content": "You are a senior telecom retention analyst."},
            {"role": "user", "content": f"{context}\n\nQuestion: {question}"}
        ],
        temperature=0.4,
        max_tokens=400
    )

    return res.choices[0].message.content.strip()


def fallback_answer(summary):
    top = summary["segments"].iloc[0]
    return (
        f"Highest risk segment: {top['risk_band']} "
        f"({int(top['customers'])} customers, "
        f"avg risk {top['avg_risk']*100:.1f}%). "
        "Focus retention efforts here first."
    )


# ------------------------------------------------
# Train once (hidden)
# ------------------------------------------------
if not os.path.exists(TRAIN_PATH):
    st.error("Training file missing: data/final_data.csv")
    st.stop()

train_df = load_csv(TRAIN_PATH)
model, features = train_model(train_df)

# ------------------------------------------------
# Upload
# ------------------------------------------------
st.sidebar.header("Data settings")
uploaded = st.sidebar.file_uploader("Upload customer CSV", type="csv")

scored_df = None
summary = None

if uploaded:
    user_df = load_csv(io.BytesIO(uploaded.read()))
    scored_df = score(user_df, model, features)
    summary = build_summary(scored_df)
    st.sidebar.success("Dataset uploaded and analysed")
else:
    st.sidebar.info("Upload a dataset to begin")

# ------------------------------------------------
# Tabs
# ------------------------------------------------
tab_chat, tab_data, tab_how = st.tabs(["Chat assistant", "Data preview", "How it works"])

with tab_chat:
    if scored_df is None:
        st.info("Upload data to enable the assistant.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Ask about churn, segments, or retention strategy")
    if q:
        st.session_state.messages.append({"role": "user", "content": q})
        try:
            answer = ask_llm(q, summary)
        except Exception:
            answer = fallback_answer(summary)

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


with tab_data:
    if scored_df is None:
        st.info("Upload data to see results.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", summary["customers"])
    c2.metric("Avg churn risk", f"{summary['avg_risk']*100:.1f}%")
    c3.metric("Top 10% threshold", f"{summary['p90']*100:.1f}%")

    st.dataframe(scored_df.head(25), use_container_width=True)
    st.dataframe(summary["segments"], use_container_width=True)


with tab_how:
    st.markdown("""
**How it works**

• The app learns churn patterns from internal data  
• Your uploaded file is analysed separately  
• Customers are grouped by churn risk  
• AI explains insights and actions  
• If AI is unavailable, the app still responds  

No training data is ever shown to users.
""")
