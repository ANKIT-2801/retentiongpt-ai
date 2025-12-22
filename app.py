import os
import io

import streamlit as st
import pandas as pd
import numpy as np

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
st.caption("Train a churn model on telco data, score customers, and ask an AI assistant for retention strategy.")

DATA_DIR = "data"
TRAIN_PATH = os.path.join(DATA_DIR, "final_data.csv")


# ------------------------------------------------
# Helper functions
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
        raise ValueError("Dataset must contain a 'churn_flag' column.")

    y = df["churn_flag"].astype(int)

    # Drop obvious leakage / ID columns if present
    drop_cols = [
        "churn_flag",
        "predicted_churn_proba",
        "predicted_ltv",
        "predicted_remaining_months",
        "customer_id",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    return X, y, num_cols, cat_cols


@st.cache_resource(show_spinner=True)
def train_churn_model(df: pd.DataFrame):
    """Train a simple, robust churn model from the training dataset."""
    X, y, num_cols, cat_cols = split_features_target(df)

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


def score_dataset(df: pd.DataFrame, model, feature_cols):
    """Apply trained model to any compatible dataset and add churn proba + risk band."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    X = df[[c for c in feature_cols if c in df.columns]].copy()

    if "churn_flag" in X.columns:
        X = X.drop(columns=["churn_flag"])

    proba = model.predict_proba(X)[:, 1]
    df["predicted_churn_proba"] = proba

    # Create 4 risk bands: Low / Medium / High / Very high
    try:
        df["risk_band"] = pd.qcut(
            df["predicted_churn_proba"],
            q=4,
            labels=["Low risk", "Medium risk", "High risk", "Very high risk"]
        )
    except ValueError:
        # Fallback if not enough unique values
        df["risk_band"] = pd.cut(
            df["predicted_churn_proba"],
            bins=[-0.01, 0.25, 0.5, 0.75, 1.0],
            labels=["Low risk", "Medium risk", "High risk", "Very high risk"]
        )

    return df


def build_context(scored_df: pd.DataFrame):
    ctx = {}

    if "predicted_churn_proba" not in scored_df.columns:
        return ctx

    probas = scored_df["predicted_churn_proba"]

    ctx["avg_risk"] = float(np.nanmean(probas))
    ctx["p90_risk"] = float(np.nanpercentile(probas, 90))
    ctx["top_risk_count"] = int((probas >= np.nanpercentile(probas, 90)).sum())
    ctx["n_customers"] = len(scored_df)

    if "risk_band" in scored_df.columns:
        seg_summary = (
            scored_df.groupby("risk_band", dropna=False)["predicted_churn_proba"]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={"count": "customers", "mean": "avg_churn_risk"})
            .sort_values("avg_churn_risk", ascending=False)
        )
        ctx["segment_summary"] = seg_summary
    else:
        ctx["segment_summary"] = None

    return ctx


def detect_intent(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["why", "reason", "increase", "decrease", "driver", "cause"]):
        return "churn_explain"
    if any(k in t for k in ["segment", "cohort", "group", "prioritise", "prioritize", "which segment"]):
        return "segment_strategy"
    if any(k in t for k in ["experiment", "test", "a/b", "ab test", "hypothesis"]):
        return "experiments"
    if any(k in t for k in ["revenue", "ltv", "value", "at risk", "risk"]):
        return "revenue"
    return "general"


def parse_rank(text: str) -> int:
    """Detect '2nd', 'second', 'third' etc. Default is 1 (top segment)."""
    t = text.lower()
    if "second" in t or "2nd" in t:
        return 2
    if "third" in t or "3rd" in t:
        return 3
    if "fourth" in t or "4th" in t:
        return 4
    return 1


def format_pct(x):
    return f"{x * 100:.1f}%"


def assistant_response(question: str, intent: str, ctx: dict) -> str:
    """Rule-based backup assistant (used if LLM fails)."""
    avg_risk = ctx.get("avg_risk")
    p90_risk = ctx.get("p90_risk")
    top_risk_count = ctx.get("top_risk_count")
    n_customers = ctx.get("n_customers")
    seg_summary = ctx.get("segment_summary")

    if not ctx:
        return "I wasn't able to build context from the data. Please check that the file has a 'churn_flag' column and re-run."

    if intent == "revenue":
        if avg_risk is None:
            return "I can’t estimate revenue risk without churn scores, but the model should have created 'predicted_churn_proba'."
        return (
            f"In this dataset I see **{n_customers:,} customers**.\n\n"
            f"- Average churn risk: **{format_pct(avg_risk)}**\n"
            f"- Top 10% churn risk threshold: **{format_pct(p90_risk)}**\n"
            f"- Customers in that top-risk bucket: **{top_risk_count:,}**\n\n"
            "You’d normally multiply those probabilities by ARPU or LTV to estimate revenue at risk."
        )

    if intent == "segment_strategy":
        if seg_summary is None or seg_summary.empty:
            return "I couldn’t create risk bands. Once 'predicted_churn_proba' is available, I’ll group customers into Low / Medium / High / Very high risk segments."

        rank = parse_rank(question)
        if rank > len(seg_summary):
            rank = len(seg_summary)

        seg_row = seg_summary.iloc[rank - 1]
        seg_name = str(seg_row["risk_band"])
        avg_seg_risk = seg_row["avg_churn_risk"]
        seg_customers = int(seg_row["customers"])

        rank_label = {1: "highest", 2: "second-highest", 3: "third-highest"}.get(rank, f"{rank}th")

        return (
            f"The **{rank_label} risk segment** is **{seg_name}**.\n\n"
            f"- Customers in this segment: **{seg_customers:,}**\n"
            f"- Average churn risk: **{format_pct(avg_seg_risk)}**\n\n"
            "Suggested focus:\n"
            "1. Review this segment’s profile in your dashboard (tenure, plan, usage, support tickets).\n"
            "2. Design targeted retention offers for this group.\n"
            "3. Track churn over the next 30–60 days to see if the risk actually drops."
        )

    if intent == "experiments":
        return (
            "Here are three simple, data-driven experiments you can run based on churn scores:\n\n"
            "1) **Top-risk outreach**\n"
            "   - Target: Customers in the 'Very high risk' band.\n"
            "   - Action: Proactive outreach (email/SMS/call) with a personalised retention offer.\n"
            "   - KPI: 30-day churn rate vs a control group.\n\n"
            "2) **Support experience upgrade**\n"
            "   - Target: High-risk customers with recent support tickets.\n"
            "   - Action: Fast-track queueing or follow-up calls to close open issues.\n"
            "   - KPI: Ticket resolution time and churn rate.\n\n"
            "3) **Usage reactivation nudges**\n"
            "   - Target: Medium-risk customers whose usage has dropped.\n"
            "   - Action: In-app tips, bonus data, or reminder campaigns.\n"
            "   - KPI: change in usage and churn over 30–60 days."
        )

    if intent == "churn_explain":
        lines = []
        if avg_risk is not None:
            lines.append(f"Across all customers, the average predicted churn risk is **{format_pct(avg_risk)}**.")
            lines.append(f"The riskiest 10% of customers are above **{format_pct(p90_risk)}** churn probability.")
        if seg_summary is not None and not seg_summary.empty:
            top = seg_summary.iloc[0]
            lines.append(
                f"The most at-risk segment is **{top['risk_band']}** with "
                f"**{format_pct(top['avg_churn_risk'])}** average churn risk "
                f"across **{int(top['customers']):,} customers**."
            )
        lines.append(
            "\nTo understand *why*, you’d typically look at:\n"
            "- tenure (short vs long)\n"
            "- contract type (month-to-month vs longer-term)\n"
            "- billing method and payment issues\n"
            "- product usage and support tickets"
        )
        return "\n\n".join(lines)

    return (
        "I’ve run a churn model on your data and grouped customers into risk bands.\n\n"
        "You can ask things like:\n"
        "- *Which segment should we prioritise?*\n"
        "- *Who is in the second highest-risk segment?*\n"
        "- *How much churn risk do we see overall?*\n"
        "- *What experiments should we run to reduce churn?*"
    )


# ---------- LLM helpers (OpenRouter via OpenAI client) ----------

@st.cache_resource(show_spinner=False)
def get_llm_client():
    """Create an OpenRouter client using the API key from Streamlit secrets."""
    api_key = st.secrets.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return client


def make_context_text(ctx: dict, scored_df: pd.DataFrame, max_rows: int = 5) -> str:
    """Turn model outputs into a short text summary for the LLM."""
    lines = []

    n = ctx.get("n_customers")
    avg_risk = ctx.get("avg_risk")
    p90_risk = ctx.get("p90_risk")
    seg_summary = ctx.get("segment_summary")

    if n:
        lines.append(f"Total customers analysed: {n}")
    if avg_risk is not None:
        lines.append(f"Average predicted churn risk: {avg_risk:.3f}")
    if p90_risk is not None:
        lines.append(f"Top 10% churn risk threshold: {p90_risk:.3f}")

    if seg_summary is not None and not seg_summary.empty:
        lines.append("\nRisk band summary (band, customers, avg_churn_risk):")
        for _, row in seg_summary.head(4).iterrows():
            lines.append(
                f"- {row['risk_band']}: {int(row['customers'])} customers, avg risk {row['avg_churn_risk']:.3f}"
            )

    sample_cols = [c for c in ["customer_id", "risk_band", "predicted_churn_proba"] if c in scored_df.columns]
    if sample_cols:
        sample = scored_df[sample_cols].head(max_rows)
        lines.append("\nSample of scored customers (first rows):")
        lines.append(sample.to_string(index=False))

    return "\n".join(lines)


def ask_llm(question: str, ctx: dict, scored_df: pd.DataFrame):
    """Ask the LLM for an answer, using churn context. Returns None if LLM not available."""
    client = get_llm_client()
    if client is None:
        return None

    context_text = make_context_text(ctx, scored_df)

    try:
        completion = client.chat.completions.create(
            model="openrouter/auto",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior customer retention and growth analyst. "
                        "You are given churn model outputs for a telecom dataset. "
                        "Use ONLY the provided context and general telco knowledge. "
                        "Be concise, structured, and business-focused."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Here is the context from the churn model and risk bands:\n\n"
                        f"{context_text}\n\n"
                        f"Now answer this question from a product/growth/CS leader:\n{question}"
                    ),
                },
            ],
            max_tokens=450,
            temperature=0.4,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"LLM call failed, falling back to rule-based assistant. Details: {e}")
        return None


# ------------------------------------------------
# Load base training data and train the model
# ------------------------------------------------
if not file_exists(TRAIN_PATH):
    st.error(f"Training file not found at `{TRAIN_PATH}`. Please add final_data.csv to the data/ folder.")
    st.stop()

base_df = load_training_data(TRAIN_PATH)
model, feature_cols = train_churn_model(base_df)

# ------------------------------------------------
# Sidebar – upload new data
# ------------------------------------------------
st.sidebar.header("Data settings")
st.sidebar.write("By default, the assistant uses the built-in training data.")
st.sidebar.write("You can optionally upload a new telco CSV with the same columns to score a fresh dataset.")

uploaded_file = st.sidebar.file_uploader("Upload new customer data (CSV)", type=["csv"])

if uploaded_file is not None:
    raw_bytes = uploaded_file.read()
    uploaded_df = pd.read_csv(io.BytesIO(raw_bytes))
    uploaded_df.columns = [c.strip().lower().replace(" ", "_") for c in uploaded_df.columns]
    scored_df = score_dataset(uploaded_df, model, feature_cols)
    st.sidebar.success("Using uploaded data for analysis.")
else:
    scored_df = score_dataset(base_df, model, feature_cols)
    st.sidebar.info("Using default training data.")

ctx = build_context(scored_df)

# ------------------------------------------------
# Main layout – tabs
# ------------------------------------------------
tab_chat, tab_data, tab_how = st.tabs(["Chat assistant", "Data preview", "How it works"])

with tab_chat:
    st.subheader("Ask questions about churn, risk bands, and retention strategy")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Type your question here...")
    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})

        # 1) Try LLM answer
        llm_reply = ask_llm(user_q, ctx, scored_df)

        if llm_reply:
            reply = llm_reply
        else:
            # 2) Fallback to rule-based answer if LLM not available
            intent = detect_intent(user_q)
            reply = assistant_response(user_q, intent, ctx)

        with st.chat_message("assistant"):
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})

with tab_data:
    st.subheader("Scored dataset preview")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Customers analysed", f"{ctx.get('n_customers', 0):,}")
    with c2:
        if ctx.get("avg_risk") is not None:
            st.metric("Avg churn risk", format_pct(ctx["avg_risk"]))
        else:
            st.metric("Avg churn risk", "N/A")
    with c3:
        if ctx.get("p90_risk") is not None:
            st.metric("Top 10% threshold", format_pct(ctx["p90_risk"]))
        else:
            st.metric("Top 10% threshold", "N/A")

    st.write("**Sample of scored customers**")
    st.dataframe(scored_df.head(25), use_container_width=True)

    if ctx.get("segment_summary") is not None:
        st.write("**Risk-band summary**")
        st.dataframe(ctx["segment_summary"], use_container_width=True)

with tab_how:
    st.subheader("How this works (for your report & interviews)")
    st.markdown(
        """
1. **Model training**  
   The app loads `data/final_data.csv` and trains a Logistic Regression churn model in scikit-learn.  
   It treats `churn_flag` as the target and uses the remaining columns as features (after dropping IDs and any existing prediction columns).

2. **Scoring & segmentation**  
   The trained model scores each customer and creates a `predicted_churn_proba` column.  
   Customers are grouped into four named risk bands: **Low**, **Medium**, **High**, and **Very high risk**.

3. **LLM assistant**  
   The assistant summarises the churn outputs (risk bands, average risk, top 10%, sample rows) and sends that to an OpenRouter LLM.  
   The LLM responds like a senior retention analyst.  
   If the LLM isn’t available, a rule-based backup still gives reasonable answers.

4. **Uploading new data**  
   You can upload a new cleaned telco CSV in the sidebar.  
   The model scores that dataset on the fly, and the assistant’s answers are based on that new file.

This gives you a full story for your project: data → model → scoring → AI retention copilot.
"""
    )
