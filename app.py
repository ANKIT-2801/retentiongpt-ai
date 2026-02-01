import os
import io
import re

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
st.caption("Upload a telco dataset, score customers, and ask an AI assistant for retention strategy.")

DATA_DIR = "data"
TRAIN_PATH = os.path.join(DATA_DIR, "final_data.csv")

TARGET_DEFAULT = "churn_flag"

# ------------------------------------------------
# Column helpers / cleaning
# ------------------------------------------------
MISSING_TOKENS = {"", " ", "na", "n/a", "null", "none", "-", "--", "nan"}

# Common “core” business concepts we try to detect (by name first)
SYNONYMS = {
    "tenure_months": ["tenure", "tenure_months", "months_active", "customer_tenure", "months_with_company"],
    "monthly_charge": ["monthly_charge", "monthlycharges", "mrc", "monthly_fee", "monthlyamount", "monthlycost"],
    "total_charge": ["total_charge", "totalcharges", "lifetime_charge", "totalamount", "total_billed"],
    "contract": ["contract", "contract_type", "plan_contract"],
    "paymentmethod": ["paymentmethod", "payment_method", "pay_method", "payment_type"],
    "internetservice": ["internetservice", "internet_service", "net_service", "broadband"],
    "customer_id": ["customer_id", "cust_id", "subscriber_id", "account_id", "id"],
    TARGET_DEFAULT: ["churn", "churn_flag", "is_churn", "churned", "target"],
}

CORE_SIGNALS = {"tenure_months", "monthly_charge"}  # keep small + realistic

# These columns (business concepts) MUST exist (after synonym renaming) or we refuse to interpret.
REQUIRED_INTERPRET_COLS = {
    "customer_id",
    "tenure_months",
    "contract",
    "monthly_charge",
    "paymentmethod",
}



def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _looks_like_money(series: pd.Series) -> bool:
    if series.dtype != "object":
        return False
    s = series.dropna().astype(str).head(50)
    if s.empty:
        return False
    # If many values contain $ or commas or look numeric-ish with decimals
    moneyish = s.str.contains(r"[$,]") | s.str.match(r"^\s*\d+(\.\d+)?\s*$")
    return float(moneyish.mean()) > 0.7


def _to_number_safe(series: pd.Series) -> pd.Series:
    # strip $, commas
    s = series.astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


def _to_bool01(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.lower()
    true_set = {"yes", "y", "true", "1", "t"}
    false_set = {"no", "n", "false", "0", "f"}
    out = pd.Series(np.nan, index=series.index)
    out[s.isin(true_set)] = 1
    out[s.isin(false_set)] = 0
    return out


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Production-friendly cleaning:
    - normalize column names
    - normalize missing tokens to NaN
    - convert yes/no columns to 0/1 where appropriate
    - convert money-ish strings to numeric
    - drop ID-like columns (too many unique values)
    """
    df = normalize_cols(df)

    # normalize missing tokens
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()
            df.loc[df[c].str.lower().isin(MISSING_TOKENS), c] = np.nan

    # convert yes/no-like columns
    for c in df.columns:
        if df[c].dtype == "object":
            s = df[c].dropna().astype(str).str.lower()
            if s.empty:
                continue
            # if most values are yes/no style
            if float(s.isin({"yes", "no", "true", "false", "y", "n", "0", "1", "t", "f"}).mean()) > 0.85:
                df[c] = _to_bool01(df[c])

    # convert money-ish strings
    for c in df.columns:
        if _looks_like_money(df[c]):
            df[c] = _to_number_safe(df[c])

    # drop obvious ID-like columns (high-cardinality strings)
    # keep customer_id if present; we'll treat it as special later
    for c in list(df.columns):
        if c == "customer_id":
            continue
        if df[c].dtype == "object":
            nun = df[c].nunique(dropna=True)
            if nun > max(50, int(0.8 * len(df))):  # very high uniqueness
                df.drop(columns=[c], inplace=True)

    return df


# ------------------------------------------------
# Schema / matching helpers
# ------------------------------------------------
def _best_match(col: str, candidates: list[str]) -> bool:
    col_clean = re.sub(r"[^a-z0-9_]", "", col.lower())
    for cand in candidates:
        cand_clean = re.sub(r"[^a-z0-9_]", "", cand.lower())
        if col_clean == cand_clean:
            return True
    return False


def apply_synonym_renames(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename uploaded columns to our internal standard where we can.
    No UI. Name-based only.
    """
    df = normalize_cols(df)
    rename_map = {}
    for internal_name, name_list in SYNONYMS.items():
        for c in df.columns:
            if _best_match(c, name_list):
                rename_map[c] = internal_name
                break
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def split_features_target(df: pd.DataFrame, target: str = TARGET_DEFAULT):
    if target not in df.columns:
        raise ValueError(f"Dataset must contain a '{target}' column for training.")

    y = df[target].astype(int)

    drop_cols = [
        target,
        "predicted_churn_proba",
        "predicted_ltv",
        "predicted_remaining_months",
        "risk_band",
        "segment_id",
        "cluster",
        "customer_id",
    ]
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
    training_cols = set(training_feature_cols)

    match_count = len(cols.intersection(training_cols))
    match_ratio = match_count / max(1, len(training_cols))

    has_core = all(c in cols for c in CORE_SIGNALS)
    has_target = target in cols

    # Safety thresholds (tune later)
    if match_ratio >= 0.40 and has_core:
        return "pretrained"

    if has_target:
        return "train_on_upload"

    return "block"


def score_dataset_aligned(df: pd.DataFrame, model, feature_cols: list[str]) -> pd.DataFrame:
    """
    Score an uploaded dataset using the pre-trained model, safely aligned:
    - missing training cols -> NaN (imputers handle)
    - extra cols -> ignored
    """
    df = df.copy()
    df = normalize_cols(df)

    X = pd.DataFrame(index=df.index)
    missing_cols = []
    for col in feature_cols:
        if col in df.columns:
            X[col] = df[col]
        else:
            X[col] = np.nan
            missing_cols.append(col)

    proba = model.predict_proba(X)[:, 1]
    df["predicted_churn_proba"] = proba

    # risk bands
    try:
        df["risk_band"] = pd.qcut(
            df["predicted_churn_proba"],
            q=4,
            labels=["Low risk", "Medium risk", "High risk", "Very high risk"]
        )
    except ValueError:
        df["risk_band"] = pd.cut(
            df["predicted_churn_proba"],
            bins=[-0.01, 0.25, 0.5, 0.75, 1.0],
            labels=["Low risk", "Medium risk", "High risk", "Very high risk"]
        )

    return df


def train_on_upload_and_score(df: pd.DataFrame, target: str = TARGET_DEFAULT):
    """
    Train a model on the uploaded dataset itself (requires target),
    then score the same dataset.
    """
    model, feature_cols = train_churn_model(df, target=target)
    scored = score_dataset_aligned(df, model, feature_cols)
    return scored, model, feature_cols


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


def make_context_text(ctx: dict, scored_df: pd.DataFrame, max_rows: int = 5, max_cols: int = 12) -> str:
    """
    Turn model outputs into a richer text summary for the LLM.

    It automatically profiles ALL columns:
    - numeric columns: mean, min, max overall and (if available) in very high risk
    - categorical columns: top categories overall and in very high risk
    """
    lines = []

    n = ctx.get("n_customers")
    avg_risk = ctx.get("avg_risk")
    p90_risk = ctx.get("p90_risk")
    seg_summary = ctx.get("segment_summary")

    # Overall churn picture
    if n:
        lines.append(f"Total customers analysed: {n}")
    if avg_risk is not None:
        lines.append(f"Average predicted churn risk (all customers): {avg_risk:.3f}")
    if p90_risk is not None:
        lines.append(f"Top 10% churn risk threshold: {p90_risk:.3f}")

    # Risk band summary
    if seg_summary is not None and not seg_summary.empty:
        lines.append("\nRisk band summary (band, customers, avg_churn_risk):")
        for _, row in seg_summary.iterrows():
            lines.append(
                f"- {row['risk_band']}: {int(row['customers'])} customers, avg risk {row['avg_churn_risk']:.3f}"
            )

    # Prepare for column profiling
    df = scored_df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Identify very high risk subset (if available)
    vh = None
    if "risk_band" in df.columns:
        vh = df[df["risk_band"] == "Very high risk"].copy()

    # Work out numeric vs categorical columns
    ignore_cols = {"predicted_churn_proba", "risk_band"}
    all_cols = [c for c in df.columns if c not in ignore_cols]

    numeric_cols = df[all_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in all_cols if c not in numeric_cols]

    # Limit how many columns we dump into the prompt
    numeric_cols = numeric_cols[: max_cols]
    cat_cols = cat_cols[: max_cols]

    # --- Numeric column profiles ---
    if numeric_cols:
        lines.append("\nNumeric feature profiles (overall, and very high risk if available):")
        for col in numeric_cols:
            series = df[col].dropna()
            if series.empty:
                continue

            overall_mean = float(series.mean())
            overall_min = float(series.min())
            overall_max = float(series.max())

            line = f"- {col}: overall mean={overall_mean:.3f}, min={overall_min:.3f}, max={overall_max:.3f}"

            if vh is not None and col in vh.columns and not vh[col].dropna().empty:
                vh_mean = float(vh[col].dropna().mean())
                line += f"; very_high_risk_mean={vh_mean:.3f}"

            lines.append(line)

    # --- Categorical column profiles ---
    if cat_cols:
        lines.append("\nCategorical feature profiles (top categories overall and in very high risk):")
        for col in cat_cols:
            # Skip columns with too many unique values (IDs etc.)
            if df[col].nunique(dropna=True) > 20:
                continue

            vc_all = df[col].value_counts(normalize=True).head(5)
            if vc_all.empty:
                continue

            lines.append(f"\nColumn '{col}' overall distribution (top values):")
            for val, frac in vc_all.items():
                lines.append(f"- {val}: {frac * 100:.1f}% of all customers")

            if vh is not None and col in vh.columns:
                vc_vh = vh[col].value_counts(normalize=True).head(5)
                if not vc_vh.empty:
                    lines.append(f"Within VERY HIGH RISK band for '{col}':")
                    for val, frac in vc_vh.items():
                        lines.append(f"- {val}: {frac * 100:.1f}% of very high risk customers")

    # Add a small raw sample for grounding
    sample_cols = [c for c in ["customer_id", "risk_band", "predicted_churn_proba"] if c in df.columns]
    if sample_cols:
        sample = df[sample_cols].head(max_rows)
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

scored_df = None
ctx = {}

if uploaded_file is not None:
    raw_bytes = uploaded_file.read()
    uploaded_df = pd.read_csv(io.BytesIO(raw_bytes))

    # Clean + synonym rename (standardize columns early)
    uploaded_df = clean_df(uploaded_df)
    uploaded_df = apply_synonym_renames(uploaded_df)

    # HARD GATE: required interpretation columns must exist after synonym renaming
    missing = [c for c in REQUIRED_INTERPRET_COLS if c not in uploaded_df.columns]
    if missing:
        st.sidebar.error(
            "Can’t run interpretation. Your upload is missing these required business columns "
            f"(after synonym matching): {', '.join(missing)}"
        )
        st.sidebar.info(
            "Fix: rename columns so they match common names. Examples:\n"
            "- customer_id / account_id\n"
            "- tenure / tenure_months\n"
            "- contract / contract_type / plan_type\n"
            "- mrc / monthly_charge / monthlycharges\n"
            "- payment_method / billing_method"
        )
        st.stop()

    # Decide how to run (pretrained scoring vs train-on-upload)
    mode = decide_mode(uploaded_df, feature_cols, target=TARGET_DEFAULT)

    if mode == "pretrained":
        scored_df = score_dataset_aligned(uploaded_df, model, feature_cols)
        st.sidebar.success("Mode: Pretrained scoring on your upload.")

    elif mode == "train_on_upload":
        scored_df, _, _ = train_on_upload_and_score(uploaded_df, target=TARGET_DEFAULT)
        st.sidebar.success("Mode: Train-on-upload (trained on your data).")

    else:
        st.sidebar.error(
            "Dataset doesn’t match the model schema and no churn/target column was found. "
            "Add a churn column (e.g., churn_"
        )
        st.stop()

    ctx = build_context(scored_df)

else:
    st.sidebar.info("No dataset uploaded yet. Please upload a CSV to continue.")
    st.stop()

# ------------------------------------------------
# Main layout – tabs
# ------------------------------------------------
tab_chat, tab_data, tab_how = st.tabs(["Chat assistant", "Data preview", "How it works"])

with tab_chat:
    st.subheader("Ask questions about churn, risk bands, and retention strategy")
    if scored_df is None:
        st.info("Upload a dataset in the sidebar to enable the assistant.")
        st.stop()

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
    if scored_df is None:
        st.info("Upload a dataset in the sidebar to see the preview.")
        st.stop()

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
