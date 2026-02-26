import os
import io
import joblib

import streamlit as st
import pandas as pd
import numpy as np

from groq import Groq


# ------------------------------------------------
# Basic page config
# ------------------------------------------------
st.set_page_config(
    page_title="RetentionGPT – Churn Assistant",
    page_icon="📉",
    layout="wide"
)

# 👉 PASTE SESSION DEFAULTS HERE
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "col_map" not in st.session_state:
    st.session_state["col_map"] = {}
if "mapping_confirmed" not in st.session_state:
    st.session_state["mapping_confirmed"] = False
if "last_uploaded_file" not in st.session_state:
    st.session_state["last_uploaded_file"] = None

st.title("RetentionGPT – Churn & Retention Assistant")
st.caption("Upload your customer CSV, score churn risk, and ask an AI assistant for retention strategy.")
st.caption("Designed for business teams — no data science background required.")
MODEL_PATH = "models/retention_minimal_model.joblib"


# ------------------------------------------------
# Model loader
# ------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


# ------------------------------------------------
# Scoring (requires: tenure, contract, totalcharges)
# ------------------------------------------------
REQUIRED_FIELDS = {
    "tenure": "Tenure",
    "contract": "Contract",
    "totalcharges": "Total Charges",
}

def normalize_name(s: str) -> str:
    return s.strip().lower().replace(" ", "_")

def apply_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_name(c) for c in df.columns]
    rename_map = {}
    for canonical, chosen in mapping.items():
        if chosen:
            rename_map[normalize_name(chosen)] = canonical
    return df.rename(columns=rename_map)


def score_dataset(df: pd.DataFrame, model):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required = ["tenure", "contract", "totalcharges"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(
            "Missing required columns for scoring: "
            + ", ".join(missing)
            + ". Required: tenure, contract, totalcharges."
        )
        st.stop()

    # Tenure -> numeric
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")

    # TotalCharges -> numeric (strip currency/commas)
    df["totalcharges"] = (
        df["totalcharges"]
        .astype(str)
        .str.strip()
        .str.replace(r"[\$,]", "", regex=True)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce")

    # Derived feature (must match training)
    df["avg_monthly_spend"] = df["totalcharges"] / df["tenure"].clip(lower=1)

    X = df[["tenure", "totalcharges", "avg_monthly_spend", "contract"]]

    try:
        proba = model.predict_proba(X)[:, 1]
    except Exception as e:
        st.error(f"Could not score the uploaded dataset. Details: {e}")
        raise

    df["predicted_churn_proba"] = proba

    # 3 risk bands
    try:
        df["risk_band"] = pd.qcut(
            df["predicted_churn_proba"],
            q=3,
            labels=["Low risk", "Medium risk", "High risk"]
        )
    except ValueError:
        df["risk_band"] = pd.cut(
            df["predicted_churn_proba"],
            bins=[-0.01, 0.33, 0.66, 1.0],
            labels=["Low risk", "Medium risk", "High risk"]
        )

    return df


# ------------------------------------------------
# Context + assistant
# ------------------------------------------------
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
    avg_risk = ctx.get("avg_risk")
    p90_risk = ctx.get("p90_risk")
    top_risk_count = ctx.get("top_risk_count")
    n_customers = ctx.get("n_customers")
    seg_summary = ctx.get("segment_summary")

    if not ctx:
        return "I wasn't able to build context from the data."

    if intent == "revenue":
        if avg_risk is None:
            return "I can’t estimate revenue risk without churn scores."
        return (
            f"In this dataset I see **{n_customers:,} customers**.\n\n"
            f"- Average churn risk: **{format_pct(avg_risk)}**\n"
            f"- Top 10% churn risk threshold: **{format_pct(p90_risk)}**\n"
            f"- Customers in that top-risk bucket: **{top_risk_count:,}**\n\n"
            "If you have revenue/LTV columns, multiply probabilities by value to estimate revenue at risk."
        )

    if intent == "segment_strategy":
        if seg_summary is None or seg_summary.empty:
            return "I couldn’t create risk bands."

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
            "1. Prioritize outreach for High risk.\n"
            "2. Offer contract upgrades or targeted incentives.\n"
            "3. Track churn over the next 30–60 days."
        )

    if intent == "experiments":
        return (
            "Three experiments you can run:\n\n"
            "1) **High-risk outreach** (High risk)\n"
            "2) **Contract upgrade offer** (Medium risk)\n"
            "3) **Value reinforcement nudges** (Low/Medium risk)\n"
        )

    if intent == "churn_explain":
        lines = []
        if avg_risk is not None:
            lines.append(f"Average predicted churn risk is **{format_pct(avg_risk)}**.")
            lines.append(f"Top 10% of customers are above **{format_pct(p90_risk)}** churn probability.")
        if seg_summary is not None and not seg_summary.empty:
            top = seg_summary.iloc[0]
            lines.append(
                f"Most at-risk segment is **{top['risk_band']}** with "
                f"**{format_pct(top['avg_churn_risk'])}** average churn risk "
                f"across **{int(top['customers']):,} customers**."
            )
        lines.append(
            "\nTypically, churn risk is driven by:\n"
            "- shorter tenure\n"
            "- month-to-month contracts\n"
            "- higher spend relative to tenure"
        )
        return "\n\n".join(lines)

    return (
        "Ask things like:\n"
        "- Which segment should we prioritize?\n"
        "- Who is in the second highest-risk segment?\n"
        "- What experiments should we run to reduce churn?"
    )


# ------------------------------------------------
# LLM helpers (Groq)
# ------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_llm_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


def make_context_text(ctx: dict, scored_df: pd.DataFrame, max_rows: int = 5, max_cols: int = 12) -> str:
    lines = []

    n = ctx.get("n_customers")
    avg_risk = ctx.get("avg_risk")
    p90_risk = ctx.get("p90_risk")
    seg_summary = ctx.get("segment_summary")

    if n:
        lines.append(f"Total customers analysed: {n}")
    if avg_risk is not None:
        lines.append(f"Average predicted churn risk (all customers): {avg_risk:.3f}")
    if p90_risk is not None:
        lines.append(f"Top 10% churn risk threshold: {p90_risk:.3f}")

    if seg_summary is not None and not seg_summary.empty:
        lines.append("\nRisk band summary (band, customers, avg_churn_risk):")
        for _, row in seg_summary.iterrows():
            lines.append(
                f"- {row['risk_band']}: {int(row['customers'])} customers, avg risk {row['avg_churn_risk']:.3f}"
            )

    df = scored_df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    high = None
    if "risk_band" in df.columns:
        high = df[df["risk_band"] == "High risk"].copy()

    ignore_cols = {"predicted_churn_proba", "risk_band"}
    all_cols = [c for c in df.columns if c not in ignore_cols]

    numeric_cols = df[all_cols].select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    cat_cols = [c for c in all_cols if c not in numeric_cols][:max_cols]

    if numeric_cols:
        lines.append("\nNumeric feature profiles (overall, and high risk if available):")
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue

            overall_mean = float(series.mean())
            overall_min = float(series.min())
            overall_max = float(series.max())

            line = f"- {col}: overall mean={overall_mean:.3f}, min={overall_min:.3f}, max={overall_max:.3f}"

            if high is not None and col in high.columns and not high[col].dropna().empty:
                high_mean = float(high[col].dropna().mean())
                line += f"; high_risk_mean={high_mean:.3f}"

            lines.append(line)

    if cat_cols:
        lines.append("\nCategorical feature profiles (top categories overall and in high risk):")
        for col in cat_cols:
            if df[col].nunique(dropna=True) > 20:
                continue

            vc_all = df[col].value_counts(normalize=True).head(5)
            if vc_all.empty:
                continue

            lines.append(f"\nColumn '{col}' overall distribution (top values):")
            for val, frac in vc_all.items():
                lines.append(f"- {val}: {frac * 100:.1f}% of all customers")

            if high is not None and col in high.columns:
                vc_high = high[col].value_counts(normalize=True).head(5)
                if not vc_high.empty:
                    lines.append(f"Within HIGH RISK band for '{col}':")
                    for val, frac in vc_high.items():
                        lines.append(f"- {val}: {frac * 100:.1f}% of high risk customers")

    sample_cols = [c for c in ["customer_id", "risk_band", "predicted_churn_proba"] if c in df.columns]
    if sample_cols:
        sample = df[sample_cols].head(max_rows)
        lines.append("\nSample of scored customers (first rows):")
        lines.append(sample.to_string(index=False))

    return "\n".join(lines)


def ask_llm(question: str, ctx: dict, scored_df: pd.DataFrame):
    client = get_llm_client()
    if client is None:
        return None

    context_text = make_context_text(ctx, scored_df)

    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior customer retention and growth analyst. "
                        "You are given churn model outputs for a telecom-style dataset. "
                        "Use ONLY the provided context and general retention knowledge. "
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
            temperature=0.4,
            max_tokens=450,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"GROQ call failed, falling back to rule-based assistant. Details: {e}")
        return None


# ------------------------------------------------
# Load model (no runtime training)
# ------------------------------------------------
model = load_model()
if model is None:
    st.error(f"Model not found at `{MODEL_PATH}`. Upload `models/retention_minimal_model.joblib` to the repo.")
    st.stop()


# ------------------------------------------------
# Upload-first gate
# ------------------------------------------------
st.sidebar.header("Upload (required)")
st.sidebar.write("Upload a CSV. If required columns are missing, you'll map them once, then chat normally.")

uploaded_file = st.sidebar.file_uploader("Upload customer data (CSV)", type=["csv"])

if uploaded_file is None:
    st.markdown("## 👋 Welcome to RetentionGPT")
    st.caption("Designed for business teams — no data science background required.")

    st.markdown("""
RetentionGPT helps you:
- Predict which customers are likely to churn
- Segment customers into **Low / Medium / High** risk
- Ask plain-English questions and get retention recommendations
""")

    st.divider()

    st.success("""
### 🚀 Start Here

1) Upload your customer CSV (left sidebar)  
2) If required columns are missing, map them once  
3) Review the **High Risk** segment first  
4) Use the **Chat** tab to ask what to do next  

You can’t break anything — upload and explore.
""")

    st.markdown("""
### 🔄 How this works (simple view)

Upload Data → Model Scores Risk → Customers Segmented → You Decide Action  
No coding required.
""")

    st.markdown("""
### 📌 Minimum required columns

Your dataset must include (or be mappable to):
- **Tenure** (how long customer has been active)
- **Contract** (monthly, yearly, etc.)
- **Total Charges** (total amount paid)

Other columns are optional.
""")

    st.markdown("""
### ❗Important

- This app does **not** change your data — it only reads it.
- Risk bands are **relative within your uploaded dataset**, not industry benchmarks.
- Churn probability is a **ranking signal**, not a guarantee of churn.
""")

    st.markdown("""
### ✅ What you’ll get after upload

- A churn probability for each customer
- A Low / Medium / High risk label
- A quick summary + suggested actions
- A downloadable scored CSV
""")

    st.stop()

raw_bytes = uploaded_file.read()
uploaded_df_raw = pd.read_csv(io.BytesIO(raw_bytes))
uploaded_df_raw.columns = [normalize_name(c) for c in uploaded_df_raw.columns]

file_fingerprint = f"{getattr(uploaded_file, 'name', 'uploaded.csv')}::{len(raw_bytes)}"
if st.session_state["last_uploaded_file"] != file_fingerprint:
    st.session_state["last_uploaded_file"] = file_fingerprint
    st.session_state["messages"] = []
    st.session_state["col_map"] = {}
    st.session_state["mapping_confirmed"] = False

cols = uploaded_df_raw.columns.tolist()
required = ["tenure", "contract", "totalcharges"]
missing = [c for c in required if c not in cols]

# If missing and not confirmed yet -> show mapping UI and stop until confirmed
if missing and not st.session_state["mapping_confirmed"]:
    st.warning("Required columns are missing. Map your columns once to continue scoring.")

    options = ["— Select —"] + cols
    current_map = st.session_state.get("col_map", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        sel_tenure = st.selectbox(
            "Map to Tenure",
            options,
            index=options.index(current_map.get("tenure")) if current_map.get("tenure") in options else 0,
            key="map_tenure"
        )
    with col2:
        sel_contract = st.selectbox(
            "Map to Contract",
            options,
            index=options.index(current_map.get("contract")) if current_map.get("contract") in options else 0,
            key="map_contract"
        )
    with col3:
        sel_total = st.selectbox(
            "Map to Total Charges",
            options,
            index=options.index(current_map.get("totalcharges")) if current_map.get("totalcharges") in options else 0,
            key="map_totalcharges"
        )

    chosen_map = {
        "tenure": None if sel_tenure == "— Select —" else sel_tenure,
        "contract": None if sel_contract == "— Select —" else sel_contract,
        "totalcharges": None if sel_total == "— Select —" else sel_total,
    }
    st.session_state["col_map"] = chosen_map

    chosen_vals = [v for v in chosen_map.values() if v is not None]
    if len(chosen_vals) < 3:
        st.info("Select a column for Tenure, Contract, and Total Charges.")
        st.stop()

    if len(set(chosen_vals)) != 3:
        st.error("Each required field must map to a different column.")
        st.stop()

    if st.button("Confirm mapping and score", type="primary"):
        st.session_state["mapping_confirmed"] = True
        st.rerun()
    else:
        st.stop()

# If missing but confirmed -> apply mapping
if missing and st.session_state["mapping_confirmed"]:
    uploaded_df = apply_mapping(uploaded_df_raw, st.session_state["col_map"])
else:
    uploaded_df = uploaded_df_raw

# Score once per rerun, but no extra clicks required
scored_df = score_dataset(uploaded_df, model)
ctx = build_context(scored_df)



# ------------------------------------------------
# Main layout – tabs
# ------------------------------------------------
tab_chat, tab_data, tab_how = st.tabs(["Chat assistant", "Data preview", "How it works"])

with tab_chat:
    st.subheader("Ask questions about churn, risk bands, and retention strategy")
    st.markdown("""
**Try asking:**
- Which segment should we prioritize?
- Why are customers churning?
- What experiments should we run?
- What does high risk typically look like?
""")
    st.caption("Tip: Responses are based on your uploaded data and the churn model outputs.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Type your question here...")
    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})

        llm_reply = ask_llm(user_q, ctx, scored_df)

        if llm_reply:
            reply = llm_reply
        else:
            intent = detect_intent(user_q)
            reply = assistant_response(user_q, intent, ctx)

        with st.chat_message("assistant"):
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})

with tab_data:
    st.subheader("Scored dataset preview")
    with st.expander("How to interpret these results"):
        st.markdown("""
**What is churn probability?**  
Each customer receives a probability between 0 and 1 representing likelihood of churn.

**What do the risk bands mean?**
- High risk → Top third most likely to churn
- Medium risk → Middle third
- Low risk → Bottom third, relatively stable customers

**How to use this:**
- Focus retention efforts on High risk customers.
- Test incentives on Medium risk.
- Monitor and upsell Low risk customers.
""")

    # --- Metrics ---
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

    st.caption("Risk bands are relative within this uploaded dataset (top/middle/bottom third), not absolute industry benchmarks.")

    # --- Quick Insight Summary ---
    st.subheader("📊 Quick Insight Summary")

    high_count = int((scored_df["risk_band"] == "High risk").sum())
    total = int(len(scored_df))

    avg_risk_val = ctx.get("avg_risk")
    avg_risk_text = format_pct(avg_risk_val) if avg_risk_val is not None else "N/A"

    st.markdown(f"""
- **{high_count:,} out of {total:,} customers** are in the **High Risk** segment.
- Average churn probability across the dataset: **{avg_risk_text}**
- Recommended next step: start with the **Top High Risk Customers** list below and prioritize outreach.
""")

    st.caption("High Risk = Needs Attention | Medium Risk = Monitor | Low Risk = Stable")

    # --- Sample preview ---
    st.write("**Sample of scored customers**")
    st.dataframe(scored_df.head(25), use_container_width=True)

    # --- Risk summary ---
    if ctx.get("segment_summary") is not None:
        st.write("**Customer Risk Breakdown**")
        st.dataframe(ctx["segment_summary"], use_container_width=True)

    # --- Prioritization table ---
    st.subheader("Who to prioritize first (Top High Risk Customers)")

    show_cols = [c for c in ["customer_id", "tenure", "contract", "totalcharges", "predicted_churn_proba", "risk_band"] if c in scored_df.columns]

    st.dataframe(
        scored_df.sort_values("predicted_churn_proba", ascending=False).head(25)[show_cols],
        use_container_width=True
    )

    # --- Recommended actions ---
    st.subheader("Recommended Actions by Segment")

    st.markdown("""
**High Risk**
- Immediate outreach
- Contract upgrade offers
- Targeted retention incentives

**Medium Risk**
- Monitor engagement
- Run small retention experiments
- Improve onboarding or feature adoption

**Low Risk**
- Maintain experience
- Upsell / cross-sell
- Loyalty programs
""")

    st.download_button(
        "Download scored customers (CSV)",
        data=scored_df.to_csv(index=False),
        file_name="scored_customers.csv",
        mime="text/csv"
    )
with tab_how:
    st.subheader("How this works (for your report & interviews)")
    st.markdown(
        """
1. **Model**  
   The app loads a pre-trained churn model from `models/retention_minimal_model.joblib`.

2. **Upload-first**  
   The app shows nothing until a CSV is uploaded.

3. **Required columns**  
   Your uploaded CSV must include:
   - `tenure` (numeric)
   - `contract` (categorical)
   - `totalcharges` (numeric / currency-like)

4. **Scoring & segmentation**  
   The app creates:
   - `predicted_churn_proba`
   - `risk_band` (Low / Medium / High)

5. **Assistant**  
   Optional Groq LLM answers questions using a data-driven context. If unavailable, a rule-based assistant responds.

6. **How to interpret results**  
   Churn probability is a ranking signal for prioritization, not a guarantee of churn.

7. **Best practice**  
   Use high-risk customers for targeted outreach and measure retention uplift over time.

"""
    )
