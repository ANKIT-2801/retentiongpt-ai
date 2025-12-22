import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Config
# ----------------------------
st.set_page_config(
    page_title="RetentionGPT",
    page_icon="📉",
    layout="wide"
)

DATA_DIR = "data"
OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")
MODEL_PATH = os.path.join(DATA_DIR, "model", "gradient_boosting_pipeline.joblib")

PATHS = {
    "final_data": os.path.join(DATA_DIR, "final_data.csv"),
    "tableau_dataset": os.path.join(DATA_DIR, "tableau_dataset.csv"),
    "analytical_with_predictions": os.path.join(OUTPUTS_DIR, "analytical_with_predictions.csv"),
    "segmented_customers": os.path.join(OUTPUTS_DIR, "segmented_customers.csv"),
    "model_auc_summary": os.path.join(OUTPUTS_DIR, "model_auc_summary.csv"),
}

# ----------------------------
# Helpers
# ----------------------------
def file_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)

def detect_intent(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["why", "reason", "increase", "decrease", "drivers", "cause"]):
        return "churn_explain"
    if any(k in t for k in ["segment", "cohort", "group", "prioritize", "priority"]):
        return "segment_strategy"
    if any(k in t for k in ["experiment", "a/b", "ab test", "test idea", "hypothesis"]):
        return "experiments"
    if any(k in t for k in ["revenue", "ltv", "impact", "at risk", "risk"]):
        return "revenue"
    if any(k in t for k in ["predict", "score", "probability", "proba", "risk score"]):
        return "predict"
    return "general"

def safe_get(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols_present = [c for c in cols if c in df.columns]
    return df[cols_present].copy() if cols_present else df.copy()

def build_context(pred_df: pd.DataFrame, seg_df: pd.DataFrame) -> dict:
    ctx = {}

    # churn probability column
    proba_col = None
    for c in ["predicted_churn_proba", "churn_probability", "churn_proba"]:
        if c in pred_df.columns:
            proba_col = c
            break

    # ltv column
    ltv_col = None
    for c in ["predicted_ltv", "ltv", "customer_ltv"]:
        if c in pred_df.columns:
            ltv_col = c
            break

    # segment column
    seg_col = "segment_id" if "segment_id" in pred_df.columns else ("segment_id" if "segment_id" in seg_df.columns else None)

    # Build summaries
    if proba_col:
        ctx["avg_risk"] = float(np.nanmean(pred_df[proba_col]))
        ctx["p90_risk"] = float(np.nanpercentile(pred_df[proba_col], 90))
        ctx["top_risk_count"] = int((pred_df[proba_col] >= np.nanpercentile(pred_df[proba_col], 90)).sum())
    else:
        ctx["avg_risk"] = None
        ctx["p90_risk"] = None
        ctx["top_risk_count"] = None

    if ltv_col:
        ctx["total_ltv"] = float(np.nansum(pred_df[ltv_col]))
        ctx["avg_ltv"] = float(np.nanmean(pred_df[ltv_col]))
    else:
        ctx["total_ltv"] = None
        ctx["avg_ltv"] = None

    # Segment risk summary
    if seg_col and proba_col and seg_col in pred_df.columns:
        seg_summary = (
            pred_df.groupby(seg_col, dropna=False)[proba_col]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={"count": "customers", "mean": "avg_churn_risk"})
            .sort_values("avg_churn_risk", ascending=False)
        )
        ctx["segment_summary"] = seg_summary
    else:
        ctx["segment_summary"] = None

    return ctx

def format_money(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"${x:,.0f}"

def assistant_response(user_q: str, intent: str, ctx: dict, auc_df: pd.DataFrame | None) -> str:
    # Keep responses short, business-readable, and grounded in your computed metrics
    avg_risk = ctx.get("avg_risk")
    p90_risk = ctx.get("p90_risk")
    top_risk_count = ctx.get("top_risk_count")
    total_ltv = ctx.get("total_ltv")
    avg_ltv = ctx.get("avg_ltv")
    seg_summary = ctx.get("segment_summary")

    auc_text = ""
    if auc_df is not None and len(auc_df.columns) >= 1:
        # Your model_auc_summary.csv might be a single column or a series-like file
        try:
            # Try to read as Series-like: index/value
            if auc_df.shape[1] == 1:
                # If it's a single column file, just show sample
                auc_text = ""
            else:
                auc_text = ""
        except:
            auc_text = ""

    if intent == "revenue":
        parts = []
        if total_ltv is not None:
            parts.append(f"Total predicted customer value in the scored dataset is **{format_money(total_ltv)}**.")
        if avg_ltv is not None:
            parts.append(f"Average predicted value per customer is **{format_money(avg_ltv)}**.")
        if avg_risk is not None:
            parts.append(f"Average churn risk across customers is **{avg_risk:.2f}**.")
        parts.append("If you want, ask: *Which segment has the highest risk?* or *What are the top actions to reduce churn?*")
        return "\n\n".join(parts)

    if intent == "segment_strategy":
        if seg_summary is None or seg_summary.empty:
            return "I can summarize segment risk once `segment_id` and `predicted_churn_proba` are present in `analytical_with_predictions.csv`."
        top = seg_summary.iloc[0]
        return (
            f"The highest-risk segment is **Segment {top.iloc[0]}** with an average churn risk of **{top['avg_churn_risk']:.2f}** "
            f"across **{int(top['customers'])}** customers.\n\n"
            "A practical approach:\n"
            "1) Focus retention offers and outreach on this segment first.\n"
            "2) Pair that with support improvements if this segment also has higher ticket volume.\n"
            "3) Track churn rate and reactivation as the primary KPIs."
        )

    if intent == "experiments":
        # Provide grounded but generic experiments (no hallucinated numbers)
        return (
            "Here are 3 retention experiments you can run:\n\n"
            "1) **Proactive support outreach**\n"
            "   - Hypothesis: Customers with unresolved tickets churn more often.\n"
            "   - Target: High ticket_count / unresolved_ticket_count customers.\n"
            "   - KPI: 30-day churn rate, ticket resolution time.\n\n"
            "2) **Month-to-month upgrade offer**\n"
            "   - Hypothesis: Longer contracts reduce churn for mid-tenure customers.\n"
            "   - Target: Month-to-month customers with medium/high churn risk.\n"
            "   - KPI: Contract conversion rate, churn rate.\n\n"
            "3) **Usage reactivation nudges**\n"
            "   - Hypothesis: Re-engagement lowers churn among low-usage customers.\n"
            "   - Target: Low avg_sessions / low avg_active_days customers.\n"
            "   - KPI: Weekly active days, churn rate."
        )

    if intent == "predict":
        if avg_risk is None:
            return "I can’t compute churn risk summary because `predicted_churn_proba` is missing in the scored dataset."
        return (
            f"Across the scored dataset, the **average churn risk** is **{avg_risk:.2f}** and the **90th percentile risk** is **{p90_risk:.2f}**.\n\n"
            f"That means roughly **{top_risk_count}** customers fall into the highest-risk bucket (top 10%).\n\n"
            "If you want, ask: *Which segment is highest risk?* or *What actions should we take for the top-risk group?*"
        )

    if intent == "churn_explain":
        # Keep it data-backed but not hallucinated
        msg = []
        if avg_risk is not None:
            msg.append(f"Current overall churn risk level (average predicted probability) is **{avg_risk:.2f}**.")
        msg.append(
            "Common churn signals in telecom typically include:\n"
            "- Higher support volume and unresolved tickets\n"
            "- Lower product usage/engagement\n"
            "- Month-to-month contracts and certain payment methods\n\n"
            "Next step: compare these patterns across segments to pinpoint which group is driving risk."
        )
        if seg_summary is not None and not seg_summary.empty:
            top = seg_summary.iloc[0]
            msg.append(f"Right now, **Segment {top.iloc[0]}** has the highest average risk (**{top['avg_churn_risk']:.2f}**).")
        return "\n\n".join(msg)

    # general
    return (
        "I can help with:\n"
        "- churn risk summary (predicted probabilities)\n"
        "- segment prioritization\n"
        "- revenue / LTV impact\n"
        "- retention experiment ideas\n\n"
        "Try asking: *Which segment should we prioritize?* or *How much revenue is at risk?*"
    )

# ----------------------------
# Load resources and data
# ----------------------------
st.title("RetentionGPT – AI Retention Assistant")
st.caption("Uses your curated churn outputs to generate retention insights. Deployed on Streamlit Cloud.")

missing = [name for name, path in PATHS.items() if not file_exists(path)]
if missing:
    st.error(
        "Some required files are missing. Please check your repo paths:\n\n"
        + "\n".join([f"- {k}: {PATHS[k]}" for k in missing])
    )
    st.stop()

if not file_exists(MODEL_PATH):
    st.warning("Model file is missing. Add: data/model/gradient_boosting_pipeline.joblib")
    # App can still run in insight-only mode if needed
    model = None
else:
    model = load_model(MODEL_PATH)

final_df = load_csv(PATHS["final_data"])
tableau_df = load_csv(PATHS["tableau_dataset"])
pred_df = load_csv(PATHS["analytical_with_predictions"])
seg_df = load_csv(PATHS["segmented_customers"])
auc_df = load_csv(PATHS["model_auc_summary"])

ctx = build_context(pred_df, seg_df)

# ----------------------------
# Layout
# ----------------------------
tab1, tab2, tab3 = st.tabs(["Chat Assistant", "Data Preview", "How It Works"])

with tab1:
    st.subheader("Ask a question")
    st.write("Examples: Why is churn increasing? Which segment should we prioritize? How much revenue is at risk?")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_q = st.chat_input("Type your question here...")
    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})

        intent = detect_intent(user_q)
        response = assistant_response(user_q, intent, ctx, auc_df)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

with tab2:
    st.subheader("Quick preview")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Rows (scored)", f"{len(pred_df):,}")
    with c2:
        if ctx.get("avg_risk") is not None:
            st.metric("Avg churn risk", f"{ctx['avg_risk']:.2f}")
        else:
            st.metric("Avg churn risk", "N/A")
    with c3:
        if ctx.get("avg_ltv") is not None:
            st.metric("Avg predicted LTV", format_money(ctx["avg_ltv"]))
        else:
            st.metric("Avg predicted LTV", "N/A")

    st.write("**Scored dataset sample**")
    st.dataframe(pred_df.head(20), use_container_width=True)

    if ctx.get("segment_summary") is not None:
        st.write("**Segment risk summary**")
        st.dataframe(ctx["segment_summary"], use_container_width=True)

with tab3:
    st.subheader("How this app works")
    st.markdown(
        """
**1) Data layer**  
You created telco datasets and exported curated outputs:
- scored customers with churn probabilities and LTV
- segmentation labels
- basic model performance summary

**2) Model layer**  
A Gradient Boosting churn model (scikit-learn pipeline) is stored in:
`data/model/gradient_boosting_pipeline.joblib`

**3) Assistant layer**  
This app does not guess from raw data. It reads your curated outputs and returns:
- churn risk summaries
- segment prioritization
- revenue/LTV impact summaries
- retention experiment ideas

**Why it stays free**  
No paid APIs are used. The app runs on Streamlit Cloud Free Tier and reads static CSVs from the repo.
"""
    )
