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
from sklearn.feature_selection import mutual_info_classif

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
st.caption("Upload telecom-style data to score churn risk (when enough key fields exist) and chat for insights.")

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

    # IMPORTANT: return training feature columns in the exact order
    return model, X.columns.tolist()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().lower().replace(" ", "_") for c in out.columns]
    return out


def schema_report(upload_df: pd.DataFrame, feature_cols: list[str], core_cols: list[str]):
    """Compute coverage and missing lists."""
    present_features = [c for c in feature_cols if c in upload_df.columns]
    missing_features = [c for c in feature_cols if c not in upload_df.columns]

    present_core = [c for c in core_cols if c in upload_df.columns]
    missing_core = [c for c in core_cols if c not in upload_df.columns]

    coverage_all = len(present_features) / max(1, len(feature_cols))
    coverage_core = len(present_core) / max(1, len(core_cols))

    extra_cols = [
        c for c in upload_df.columns
        if c not in feature_cols + ["churn_flag", "predicted_churn_proba", "risk_band"]
    ]

    return {
        "present_features": present_features,
        "missing_features": missing_features,
        "coverage_all": coverage_all,
        "present_core": present_core,
        "missing_core": missing_core,
        "coverage_core": coverage_core,
        "extra_cols": extra_cols,
    }


def choose_mode(rep: dict, full_threshold: float = 0.80, limited_threshold: float = 0.50) -> str:
    """
    Mode logic:
      - FULL: all core present OR >= full_threshold of total features
      - LIMITED: core mostly present OR >= limited_threshold of total features
      - DATA_ONLY: otherwise
    """
    all_core_present = len(rep["missing_core"]) == 0
    if all_core_present or rep["coverage_all"] >= full_threshold:
        return "FULL"
    if rep["coverage_all"] >= limited_threshold or rep["coverage_core"] >= 0.60:
        return "LIMITED"
    return "DATA_ONLY"


def score_dataset(df: pd.DataFrame, model, feature_cols: list[str]):
    """
    Scores uploaded data with the trained model.
    Robust to missing columns: creates them as NaN -> imputers use training defaults.
    Extra columns are ignored for ML scoring.
    """
    df = normalize_columns(df)

    X = pd.DataFrame(index=df.index)
    missing_cols = []
    for col in feature_cols:
        if col in df.columns:
            X[col] = df[col]
        else:
            X[col] = np.nan
            missing_cols.append(col)

    if missing_cols:
        st.warning(
            f"ML scoring note: {len(missing_cols)} trained feature columns are missing. "
            "They will use training defaults during scoring. "
            f"Examples: {', '.join(missing_cols[:8])}"
            + (" ..." if len(missing_cols) > 8 else "")
        )

    # Predict churn probabilities
    proba = model.predict_proba(X)[:, 1]
    df["predicted_churn_proba"] = proba

    # Risk bands
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


def build_context(scored_df: pd.DataFrame):
    """Context for ML mode."""
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


def build_data_only_context(df: pd.DataFrame) -> dict:
    """Context for Data-Only mode (no churn scoring)."""
    df = normalize_columns(df)

    ctx = {
        "mode": "DATA_ONLY",
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": df.columns.tolist()
    }

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    # Numeric summaries (keep compact)
    num_summary = {}
    for c in numeric_cols[:25]:
        s = df[c].dropna()
        if not s.empty:
            num_summary[c] = {
                "mean": float(s.mean()),
                "min": float(s.min()),
                "max": float(s.max())
            }

    # Categorical summaries (avoid very high-cardinality cols)
    cat_summary = {}
    for c in cat_cols:
        if df[c].nunique(dropna=True) <= 30:
            vc = df[c].value_counts(normalize=True).head(5)
            if not vc.empty:
                cat_summary[c] = [(str(k), float(v)) for k, v in vc.items()]
        if len(cat_summary) >= 25:
            break

    ctx["numeric_summary"] = num_summary
    ctx["cat_summary"] = cat_summary

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


def format_pct(x):
    return f"{x * 100:.1f}%"


# ------------------------------------------------
# Core columns selection (Suggestion 2)
# ------------------------------------------------
def infer_core_columns_from_keywords(feature_cols: list[str]) -> list[str]:
    """
    Keyword-based fallback: picks likely 'core' telecom fields if present.
    """
    keywords = [
        "tenure",
        "contract",
        "monthly",
        "charges",
        "payment",
        "bill",
        "internet",
        "phone",
        "service",
        "totalcharges",
    ]
    hits = []
    for c in feature_cols:
        if any(k in c for k in keywords):
            hits.append(c)
    # Keep it small and stable
    return hits[:8]


@st.cache_data(show_spinner=False)
def infer_core_columns_mi(train_df: pd.DataFrame, k: int = 7) -> list[str]:
    """
    Auto-detect core columns from training data using mutual information.
    Works on original columns (not one-hot), so it’s easy to explain.
    """
    X, y, num_cols, cat_cols = split_features_target(train_df)

    # Basic cleaning and encoding (simple + stable)
    X2 = X.copy()

    # Fill missing
    for c in num_cols:
        X2[c] = X2[c].fillna(X2[c].median())
    for c in cat_cols:
        X2[c] = X2[c].fillna("missing").astype(str)

    # One-hot encode for MI calculation
    X_enc = pd.get_dummies(X2, columns=cat_cols, drop_first=False)

    # Mutual information needs non-negative finite
    X_enc = X_enc.replace([np.inf, -np.inf], np.nan).fillna(0)

    mi = mutual_info_classif(X_enc, y, discrete_features=False, random_state=42)
    mi_series = pd.Series(mi, index=X_enc.columns).sort_values(ascending=False)

    # Map dummy columns back to original columns
    # Example: contract_month-to-month -> contract
    def base_col(dummy_name: str) -> str:
        return dummy_name.split("_")[0] if "_" in dummy_name else dummy_name

    scores = {}
    for dummy, val in mi_series.items():
        b = base_col(dummy)
        scores[b] = max(scores.get(b, 0.0), float(val))

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    core = [name for name, _ in ranked[:k]]
    return core


# ------------------------------------------------
# LLM helpers
# ------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_llm_client():
    api_key = st.secrets.get("OPENROUTER_API_KEY")
    if not api_key:
        return None
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def make_context_text_ml(ctx: dict, scored_df: pd.DataFrame, max_rows: int = 5, max_cols: int = 12) -> str:
    """
    ML mode context for the LLM:
    - churn summary
    - risk band summary
    - automatic profiling across columns (compact)
    """
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
        for _, row in seg_summary.iterrows():
            lines.append(f"- {row['risk_band']}: {int(row['customers'])} customers, avg risk {row['avg_churn_risk']:.3f}")

    df = normalize_columns(scored_df)

    # Identify very high risk subset
    vh = None
    if "risk_band" in df.columns:
        vh = df[df["risk_band"] == "Very high risk"].copy()

    ignore_cols = {"predicted_churn_proba", "risk_band"}
    all_cols = [c for c in df.columns if c not in ignore_cols]

    numeric_cols = df[all_cols].select_dtypes(include=[np.number]).columns.tolist()[:max_cols]
    cat_cols = [c for c in all_cols if c not in numeric_cols][:max_cols]

    if numeric_cols:
        lines.append("\nNumeric feature profiles (overall, and very high risk mean if available):")
        for col in numeric_cols:
            s = df[col].dropna()
            if s.empty:
                continue
            line = f"- {col}: mean={float(s.mean()):.3f}, min={float(s.min()):.3f}, max={float(s.max()):.3f}"
            if vh is not None and col in vh.columns and not vh[col].dropna().empty:
                line += f"; very_high_risk_mean={float(vh[col].dropna().mean()):.3f}"
            lines.append(line)

    if cat_cols:
        lines.append("\nCategorical feature profiles (top values overall and in very high risk):")
        for col in cat_cols:
            if df[col].nunique(dropna=True) > 20:
                continue
            vc_all = df[col].value_counts(normalize=True).head(5)
            if vc_all.empty:
                continue
            lines.append(f"\nColumn '{col}' overall (top values):")
            for val, frac in vc_all.items():
                lines.append(f"- {val}: {frac*100:.1f}%")
            if vh is not None and col in vh.columns:
                vc_vh = vh[col].value_counts(normalize=True).head(5)
                if not vc_vh.empty:
                    lines.append(f"Within VERY HIGH RISK for '{col}':")
                    for val, frac in vc_vh.items():
                        lines.append(f"- {val}: {frac*100:.1f}%")

    sample_cols = [c for c in ["customer_id", "risk_band", "predicted_churn_proba"] if c in df.columns]
    if sample_cols:
        lines.append("\nSample scored rows:")
        lines.append(df[sample_cols].head(max_rows).to_string(index=False))

    return "\n".join(lines)


def make_context_text_data_only(ctx: dict) -> str:
    lines = []
    lines.append("MODE: DATA-ONLY (ML churn scoring disabled due to missing key trained fields).")
    lines.append(f"Rows: {ctx.get('n_rows')}, Columns: {ctx.get('n_cols')}")

    num_summary = ctx.get("numeric_summary") or {}
    cat_summary = ctx.get("cat_summary") or {}

    if num_summary:
        lines.append("\nNumeric columns summary (mean/min/max):")
        for col, stats in num_summary.items():
            lines.append(f"- {col}: mean={stats['mean']:.3f}, min={stats['min']:.3f}, max={stats['max']:.3f}")

    if cat_summary:
        lines.append("\nCategorical columns summary (top values):")
        for col, items in cat_summary.items():
            lines.append(f"\n{col}:")
            for val, frac in items:
                lines.append(f"- {val}: {frac*100:.1f}%")

    return "\n".join(lines)


def ask_llm(question: str, mode: str, ctx: dict, scored_df: pd.DataFrame | None):
    client = get_llm_client()
    if client is None:
        return None

    if mode == "DATA_ONLY":
        context_text = make_context_text_data_only(ctx)
        system_prompt = (
            "You are a senior telecom retention and growth analyst. "
            "Churn predictions are NOT available for this upload because key trained fields are missing. "
            "Answer using only the provided dataset summaries and general telecom reasoning. "
            "Be transparent about limitations and suggest what additional fields would unlock deeper analysis."
        )
    else:
        context_text = make_context_text_ml(ctx, scored_df)
        system_prompt = (
            "You are a senior customer retention and growth analyst. "
            "You are given churn model outputs for a telecom dataset. "
            "Use ONLY the provided context and general telco knowledge. "
            "Be concise, structured, and business-focused."
        )

        if mode == "LIMITED":
            system_prompt += " Note: the dataset is missing several trained features, so scoring may be less reliable."

    try:
        completion = client.chat.completions.create(
            model="openrouter/auto",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n\n{context_text}\n\nQuestion:\n{question}"},
            ],
            max_tokens=550,
            temperature=0.4,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"LLM call failed. Details: {e}")
        return None


# ------------------------------------------------
# Load base training data and train model
# ------------------------------------------------
if not file_exists(TRAIN_PATH):
    st.error(f"Training file not found at `{TRAIN_PATH}`. Please add final_data.csv to the data/ folder.")
    st.stop()

base_df = load_training_data(TRAIN_PATH)
model, feature_cols = train_churn_model(base_df)

# Core columns (auto)
core_cols_auto = infer_core_columns_mi(base_df, k=7)

# If MI picks something odd (rare), fall back to keywords
if len(core_cols_auto) < 4:
    core_cols_auto = infer_core_columns_from_keywords(feature_cols)
    if len(core_cols_auto) < 4:
        core_cols_auto = feature_cols[:7]

# ------------------------------------------------
# Sidebar
# ------------------------------------------------
st.sidebar.header("Data settings")
st.sidebar.write("Upload a telecom-style CSV. The app adapts based on how complete the input is.")

# Allow override if you want
use_custom_core = st.sidebar.checkbox("Override core columns (advanced)", value=False)
if use_custom_core:
    core_cols = st.sidebar.multiselect(
        "Select core columns required for churn scoring",
        options=feature_cols,
        default=core_cols_auto
    )
    if len(core_cols) < 3:
        st.sidebar.warning("Pick at least 3 core columns to make this meaningful.")
else:
    core_cols = core_cols_auto

st.sidebar.caption(f"Core columns currently used: {', '.join(core_cols)}")

uploaded_file = st.sidebar.file_uploader("Upload new customer data (CSV)", type=["csv"])

# ------------------------------------------------
# Process dataset + mode selection
# ------------------------------------------------
mode = "FULL"
rep = None

if uploaded_file is not None:
    raw_bytes = uploaded_file.read()
    uploaded_df = pd.read_csv(io.BytesIO(raw_bytes))
    uploaded_df = normalize_columns(uploaded_df)

    rep = schema_report(uploaded_df, feature_cols, core_cols)
    mode = choose_mode(rep, full_threshold=0.80, limited_threshold=0.50)

    # Show a clear status (this is the UX part that prevents “it’s broken”)
    st.sidebar.markdown("### Input quality check")
    st.sidebar.write(f"Feature coverage: **{rep['coverage_all']*100:.0f}%**")
    st.sidebar.write(f"Core coverage: **{rep['coverage_core']*100:.0f}%**")
    st.sidebar.write(f"Mode: **{mode}**")

    if rep["missing_core"]:
        st.sidebar.warning("Missing core fields: " + ", ".join(rep["missing_core"][:8]) + (" ..." if len(rep["missing_core"]) > 8 else ""))
    if rep["extra_cols"]:
        st.sidebar.info("Extra columns detected (ignored by model): " + ", ".join(rep["extra_cols"][:8]) + (" ..." if len(rep["extra_cols"]) > 8 else ""))

    if mode == "DATA_ONLY":
        scored_df = uploaded_df.copy()
        ctx = build_data_only_context(scored_df)
        st.sidebar.warning("Data-Only mode: churn scoring is disabled for this upload.")
    else:
        scored_df = score_dataset(uploaded_df, model, feature_cols)
        ctx = build_context(scored_df)
        ctx["mode"] = mode

        if mode == "LIMITED":
            st.sidebar.warning("Limited mode: churn scoring runs, but missing fields may reduce reliability.")
        else:
            st.sidebar.success("Full mode: churn scoring enabled.")
else:
    # Default training dataset
    scored_df = score_dataset(base_df, model, feature_cols)
    ctx = build_context(scored_df)
    ctx["mode"] = "FULL"
    st.sidebar.info("Using default training data (Full mode).")

# ------------------------------------------------
# UI Tabs
# ------------------------------------------------
tab_chat, tab_data, tab_how = st.tabs(["Chat assistant", "Data preview", "How it works"])

with tab_chat:
    st.subheader("Ask questions about churn, risk bands, and retention strategy")

    if mode == "DATA_ONLY":
        st.info(
            "This upload is missing key churn fields, so **predictions are disabled**. "
            "You can still ask questions and get descriptive insights. "
            "For churn scoring, upload more of the core telecom fields."
        )
    elif mode == "LIMITED":
        st.warning(
            "Churn scoring is enabled, but the upload is missing several trained fields. "
            "Treat predictions as directional and upload more core fields for stronger results."
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Type your question here...")
    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})

        llm_reply = ask_llm(user_q, mode, ctx, None if mode == "DATA_ONLY" else scored_df)

        if llm_reply:
            reply = llm_reply
        else:
            # Fallback if no LLM key
            if mode == "DATA_ONLY":
                reply = (
                    "I can’t use the AI assistant right now (missing API key or request failed). "
                    "But I can still summarise the dataset in the Data Preview tab."
                )
            else:
                # basic fallback summary
                reply = (
                    f"Customers analysed: {ctx.get('n_customers', 0):,}\n\n"
                    f"Avg churn risk: {format_pct(ctx.get('avg_risk', 0.0))} (if scoring enabled)"
                )

        with st.chat_message("assistant"):
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})

with tab_data:
    st.subheader("Dataset preview")

    if mode != "DATA_ONLY":
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Customers analysed", f"{ctx.get('n_customers', 0):,}")
        with c2:
            st.metric("Avg churn risk", format_pct(ctx["avg_risk"]) if ctx.get("avg_risk") is not None else "N/A")
        with c3:
            st.metric("Top 10% threshold", format_pct(ctx["p90_risk"]) if ctx.get("p90_risk") is not None else "N/A")

        if ctx.get("segment_summary") is not None:
            st.write("**Risk-band summary**")
            st.dataframe(ctx["segment_summary"], use_container_width=True)

    st.write("**Sample rows**")
    st.dataframe(scored_df.head(25), use_container_width=True)

with tab_how:
    st.subheader("How this works")

    st.markdown(
        """
**RetentionGPT runs in three modes depending on upload completeness:**

- **Full Mode:** Enough core fields (or ~80% of trained fields).  
  The model scores churn risk and creates risk bands.

- **Limited Mode:** Partial coverage (~50–80% or partial core).  
  The model still scores, but predictions are less reliable.

- **Data-Only Mode:** Too many missing core fields.  
  Predictions are disabled, but the assistant still provides descriptive insights from the uploaded data and suggests what fields to add for deeper churn analysis.

This design prevents the product from feeling “broken” when users upload incomplete datasets.
"""
    )
