import os
import streamlit as st
import pandas as pd
import numpy as np

# ----------------------------
# Basic page config
# ----------------------------
st.set_page_config(
    page_title="RetentionGPT",
    page_icon="📉",
    layout="wide"
)

# ----------------------------
# File paths (based on your repo)
# ----------------------------
DATA_DIR = "data"
OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")

PATHS = {
    "tableau_dataset": os.path.join(DATA_DIR, "tableau_dataset.csv"),
    "analytical_with_predictions": os.path.join(OUTPUTS_DIR, "analytical_with_predictions.csv"),
    "segmented_customers": os.path.join(OUTPUTS_DIR, "segmented_customers.csv"),
    "model_auc_summary": os.path.join(OUTPUTS_DIR, "model_auc_summary.csv"),
}


# ----------------------------
# Helper functions
# ----------------------------
def file_exists(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # standardise column names a bit
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def detect_intent(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["why", "increase", "decrease", "reason", "cause", "driver"]):
        return "churn_explain"
    if any(k in t for k in ["segment", "cohort", "group", "prioritize", "priority"]):
        return "segment_strategy"
    if any(k in t for k in ["experiment", "test", "a/b", "ab test", "hypothesis"]):
        return "experiments"
    if any(k in t for k in ["revenue", "ltv", "value", "at risk", "risk"]):
        return "revenue"
    return "general"


def build_context(pred_df: pd.DataFrame, seg_df: pd.DataFrame | None):
    ctx = {}

    # find churn probability column
    proba_col = None
    for c in ["predicted_churn_proba", "churn_proba", "churn_probability"]:
        if c in pred_df.columns:
            proba_col = c
            break

    # find LTV column
    ltv_col = None
    for c in ["predicted_ltv", "ltv", "customer_ltv"]:
        if c in pred_df.columns:
            ltv_col = c
            break

    # segment column
    seg_col = None
    if "segment_id" in pred_df.columns:
        seg_col = "segment_id"
    elif seg_df is not None and "segment_id" in seg_df.columns:
        seg_col = "segment_id"

    # churn risk stats
    if proba_col:
        probas = pred_df[proba_col]
        ctx["avg_risk"] = float(np.nanmean(probas))
        ctx["p90_risk"] = float(np.nanpercentile(probas, 90))
        ctx["top_risk_count"] = int((probas >= np.nanpercentile(probas, 90)).sum())
    else:
        ctx["avg_risk"] = ctx["p90_risk"] = None
        ctx["top_risk_count"] = None

    # LTV stats
    if ltv_col:
        ctx["total_ltv"] = float(np.nansum(pred_df[ltv_col]))
        ctx["avg_ltv"] = float(np.nanmean(pred_df[ltv_col]))
    else:
        ctx["total_ltv"] = ctx["avg_ltv"] = None

    # segment summary
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


def assistant_response(user_q: str, intent: str, ctx: dict) -> str:
    avg_risk = ctx.get("avg_risk")
    p90_risk = ctx.get("p90_risk")
    top_risk_count = ctx.get("top_risk_count")
    total_ltv = ctx.get("total_ltv")
    avg_ltv = ctx.get("avg_ltv")
    seg_summary = ctx.get("segment_summary")

    if intent == "revenue":
        parts = []
        if total_ltv is not None:
            parts.append(f"Total predicted customer value in the scored dataset is **{format_money(total_ltv)}**.")
        if avg_ltv is not None:
            parts.append(f"Average predicted value per customer is **{format_money(avg_ltv)}**.")
        if avg_risk is not None:
            parts.append(f"Average churn risk across customers is **{avg_risk:.2f}**.")
        parts.append("You can ask: *Which segment should we focus on first?* for a more targeted view.")
        return "\n\n".join(parts)

    if intent == "segment_strategy":
        if seg_summary is None or seg_summary.empty:
            return "Once `segment_id` and `predicted_churn_proba` are present in the scored data, I can summarise which segment is highest risk."
        top = seg_summary.iloc[0]
        return (
            f"The highest-risk segment is **Segment {top.iloc[0]}** with an average churn risk of **{top['avg_churn_risk']:.2f}** "
            f"across **{int(top['customers'])}** customers.\n\n"
            "A sensible strategy:\n"
            "1) Prioritise outreach and offers for this segment.\n"
            "2) Review their support and usage patterns in your dashboard.\n"
            "3) Track churn rate and reactivation as the main KPIs."
        )

    if intent == "experiments":
        return (
            "Here are 3 practical retention experiments you could run:\n\n"
            "1) **Proactive support outreach**\n"
            "   - Target customers with high ticket_count or unresolved tickets.\n"
            "   - KPI: 30-day churn rate and resolution time.\n\n"
            "2) **Contract / plan upgrade offer**\n"
            "   - Target month-to-month or high-risk customers.\n"
            "   - KPI: upgrade rate and churn rate.\n\n"
            "3) **Usage reactivation nudges**\n"
            "   - Target low-usage customers (low sessions or active days).\n"
            "   - KPI: weekly active days and churn rate."
        )

    if intent == "churn_explain":
        msg = []
        if avg_risk is not None:
            msg.append(f"Overall average predicted churn risk is **{avg_risk:.2f}**.")
        msg.append(
            "In the telco context, churn is usually driven by a mix of support issues, low usage, and contract type.\n\n"
            "Your dashboard can be used to compare churn risk across segments, plans, and payment methods to see where the pressure is highest."
        )
        if seg_summary is not None and not seg_summary.empty:
            top = seg_summary.iloc[0]
            msg.append(f"Right now, **Segment {top.iloc[0]}** looks like the most at-risk group.")
        return "\n\n".join(msg)

    # general / fallback
    return (
        "I can help you interpret the churn scores and segments.\n\n"
        "Try questions like:\n"
        "- *Which segment should we prioritise?*\n"
        "- *How much revenue is at risk?*\n"
        "- *Why is churn increasing?*\n"
        "- *What retention experiments can we run?*"
    )


# ----------------------------
# Load data
# ----------------------------
st.title("RetentionGPT – AI Retention Assistant")
st.caption("Uses your churn model outputs to generate retention insights. Powered by static CSVs – no paid APIs.")

missing = [name for name, path in PATHS.items() if not file_exists(path)]
if missing:
    st.error(
        "Some required files are missing in the repo:\n\n"
        + "\n".join([f"- {k}: {PATHS[k]}" for k in missing])
    )
    st.stop()

pred_df = load_csv(PATHS["analytical_with_predictions"])
seg_df = load_csv(PATHS["segmented_customers"])
tableau_df = load_csv(PATHS["tableau_dataset"])

ctx = build_context(pred_df, seg_df)

# ----------------------------
# Layout with tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["Chat assistant", "Data preview", "How it works"])

with tab1:
    st.subheader("Ask about churn, segments, or revenue at risk")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # show history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_q = st.chat_input("Type your question here...")
    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})

        intent = detect_intent(user_q)
        resp = assistant_response(user_q, intent, ctx)

        with st.chat_message("assistant"):
            st.markdown(resp)

        st.session_state.messages.append({"role": "assistant", "content": resp})

with tab2:
    st.subheader("Quick data preview")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Scored customers", f"{len(pred_df):,}")
    with c2:
        if ctx.get("avg_risk") is not None:
            st.metric("Avg churn risk", f"{ctx['avg_risk']:.2f}")
        else:
            st.metric("Avg churn risk", "N/A")
    with c3:
        if ctx.get("avg_ltv") is not None:
            st.metric("Avg predicted LTV", format_money(ctx['avg_ltv']))
        else:
            st.metric("Avg predicted LTV", "N/A")

    st.write("**Sample from analytical_with_predictions.csv**")
    st.dataframe(pred_df.head(20), use_container_width=True)

    if ctx.get("segment_summary") is not None:
        st.write("**Segment risk summary**")
        st.dataframe(ctx["segment_summary"], use_container_width=True)

with tab3:
    st.subheader("How this app works")
    st.markdown(
        """
**1. Offline modelling**

You trained churn and LTV models in a Jupyter notebook and exported:
- customer-level churn probabilities
- segment IDs
- predicted LTV and other features

Those results are stored in `data/outputs/analytical_with_predictions.csv`
and `data/outputs/segmented_customers.csv`.

**2. Analytics layer**

You built a Tableau dashboard on top of `data/tableau_dataset.csv`
to visualise churn, segments, and revenue at risk.

**3. Assistant layer**

This Streamlit app reads the exported CSVs and:
- summarises churn risk and segment performance
- explains where to focus retention efforts
- suggests simple experiment ideas

No raw database access and no paid AI APIs are used – everything runs
on Streamlit Cloud Free Tier using static files from this GitHub repo.
"""
    )
