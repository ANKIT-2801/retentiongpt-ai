import io
import os
import re
from hashlib import md5

import streamlit as st
import pandas as pd
import numpy as np
import joblib


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="RetentionGPT – Telco Churn Assistant",
    page_icon="📉",
    layout="wide"
)

st.title("RetentionGPT – Telco Churn & Retention Assistant")
st.caption("Upload your trained model + customer dataset, map key fields, score churn risk, and segment customers.")


# -----------------------------
# Constants + synonyms
# -----------------------------
TARGET_DEFAULT = "churn_flag"

MISSING_TOKENS = {"", " ", "na", "n/a", "null", "none", "-", "--", "nan"}

# Synonyms used ONLY for initial guesses in mapping dropdowns
SYNONYMS = {
    "customer_id": ["customerid", "customer_id", "cust_id", "subscriber_id", "account_id", "billing_account", "id"],
    "tenure_months": ["tenure", "tenure_months", "months_active", "months_with_company", "customer_tenure", "tenure_in_months"],
    "contract": ["contract", "contract_type", "plan_contract", "term", "plan_type", "commitment"],
    "monthly_charge": ["monthlycharges", "monthly_charge", "mrc", "monthly_fee", "monthly_cost", "monthlyamount", "recurring_charge"],
    "paymentmethod": ["paymentmethod", "payment_method", "pay_method", "payment_type", "bill_pay_type", "billing_method"],
}


# -----------------------------
# Basic helpers
# -----------------------------
def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _best_match(col: str, candidates: list[str]) -> bool:
    col_clean = re.sub(r"[^a-z0-9_]", "", str(col).lower())
    for cand in candidates:
        cand_clean = re.sub(r"[^a-z0-9_]", "", str(cand).lower())
        if col_clean == cand_clean:
            return True
    return False


def _looks_like_money(series: pd.Series) -> bool:
    if series.dtype != "object":
        return False
    s = series.dropna().astype(str).head(50)
    if s.empty:
        return False
    moneyish = s.str.contains(r"[$,]") | s.str.match(r"^\s*\d+(\.\d+)?\s*$")
    return float(moneyish.mean()) > 0.7


def _to_number_safe(series: pd.Series) -> pd.Series:
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
    Simple, reliable cleaning:
    - normalize column names
    - normalize missing tokens to NaN
    - convert yes/no style columns to 0/1
    - convert money-ish strings to numeric
    - drop obvious ID-like columns (high-cardinality strings), except we keep customer_id if mapped later
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
            if float(s.isin({"yes", "no", "true", "false", "y", "n", "0", "1", "t", "f"}).mean()) > 0.85:
                df[c] = _to_bool01(df[c])

    # convert money-ish strings
    for c in df.columns:
        if _looks_like_money(df[c]):
            df[c] = _to_number_safe(df[c])

    # drop high-cardinality object columns (IDs, free text)
    for c in list(df.columns):
        if df[c].dtype == "object":
            nun = df[c].nunique(dropna=True)
            if nun > max(60, int(0.85 * len(df))):
                df.drop(columns=[c], inplace=True)

    return df


@st.cache_resource(show_spinner=False)
def load_joblib_model_from_bytes(b: bytes):
    return joblib.load(io.BytesIO(b))


def get_expected_feature_cols(pipeline) -> list[str]:
    """
    Tries to retrieve the raw input feature names the pipeline expects.
    Works best if your pipeline has a ColumnTransformer with explicit columns.
    """
    # common pattern: Pipeline([("prep", ColumnTransformer(...)), ("clf", ...)])
    try:
        preproc = pipeline.named_steps["prep"]
        cols = []
        for _, _, c in preproc.transformers_:
            if isinstance(c, list):
                cols.extend(c)
        # de-dup keep order
        seen = set()
        ordered = []
        for x in cols:
            if x not in seen:
                seen.add(x)
                ordered.append(x)
        return ordered
    except Exception:
        pass

    # fallback: feature_names_in_ exists for some estimators/pipelines
    if hasattr(pipeline, "feature_names_in_"):
        return list(pipeline.feature_names_in_)

    return []


def align_df_to_model(df: pd.DataFrame, expected_cols: list[str]) -> pd.DataFrame:
    """
    Ensure df has all expected cols for the model. Missing -> NaN.
    """
    X = pd.DataFrame(index=df.index)
    for c in expected_cols:
        X[c] = df[c] if c in df.columns else np.nan
    return X


def risk_bands_from_proba(proba: np.ndarray) -> pd.Series:
    s = pd.Series(proba)
    try:
        return pd.qcut(s, q=4, labels=["Low risk", "Medium risk", "High risk", "Very high risk"])
    except Exception:
        return pd.cut(s, bins=[-0.01, 0.25, 0.5, 0.75, 1.0],
                      labels=["Low risk", "Medium risk", "High risk", "Very high risk"])


def build_context(scored_df: pd.DataFrame) -> dict:
    p = scored_df["predicted_churn_proba"]
    ctx = {
        "n_customers": int(len(scored_df)),
        "avg_risk": float(p.mean()),
        "p90_risk": float(np.percentile(p, 90)),
        "top_risk_count": int((p >= np.percentile(p, 90)).sum()),
    }
    ctx["segment_summary"] = (
        scored_df.groupby("risk_band")["predicted_churn_proba"]
        .agg(customers="count", avg_churn_risk="mean")
        .reset_index()
        .sort_values("avg_churn_risk", ascending=False)
    )
    return ctx


def format_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# -----------------------------
# Sidebar: upload MODEL + DATA
# -----------------------------
st.sidebar.header("Step 1: Upload model")
model_file = st.sidebar.file_uploader("Upload ML model (.joblib)", type=["joblib"])

if model_file is None:
    st.info("Upload your trained .joblib model to start.")
    st.stop()

model_bytes = model_file.read()
model_hash = md5(model_bytes).hexdigest()

model = load_joblib_model_from_bytes(model_bytes)
expected_cols = get_expected_feature_cols(model)

st.sidebar.success("Model loaded.")
if expected_cols:
    st.sidebar.caption(f"Model expects ~{len(expected_cols)} input columns.")
else:
    st.sidebar.warning("Could not infer expected input columns from the model. Scoring may fail if schema differs.")

st.sidebar.divider()
st.sidebar.header("Step 2: Upload dataset")
uploaded_file = st.sidebar.file_uploader("Upload customer data (CSV)", type=["csv"])

# Clear chat when model or dataset changes
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_key" not in st.session_state:
    st.session_state.active_key = None

data_key = None
if uploaded_file is not None:
    data_key = (uploaded_file.name, uploaded_file.size)

new_active_key = (model_hash, data_key)
if new_active_key != st.session_state.active_key:
    st.session_state.messages = []
    st.session_state.active_key = new_active_key

if uploaded_file is None:
    st.info("Upload a customer CSV to continue.")
    st.stop()

raw = uploaded_file.read()
uploaded_df = pd.read_csv(io.BytesIO(raw))
uploaded_df = clean_df(uploaded_df)

# -----------------------------
# Step 3: Required mapping UI
# -----------------------------
st.sidebar.divider()
st.sidebar.header("Step 3: Map required fields")

cols = ["(not available)"] + uploaded_df.columns.tolist()

def guess_from_synonyms(syns: list[str]) -> str:
    for c in uploaded_df.columns:
        if _best_match(c, syns):
            return c
    return "(not available)"

default_customer = guess_from_synonyms(SYNONYMS["customer_id"])
default_tenure = guess_from_synonyms(SYNONYMS["tenure_months"])
default_contract = guess_from_synonyms(SYNONYMS["contract"])
default_mrc = guess_from_synonyms(SYNONYMS["monthly_charge"])
default_payment = guess_from_synonyms(SYNONYMS["paymentmethod"])

map_customer = st.sidebar.selectbox(
    "Customer ID\nOne unique identifier for each customer/account (used to label and export results).",
    cols,
    index=cols.index(default_customer) if default_customer in cols else 0
)

map_tenure = st.sidebar.selectbox(
    "Tenure\nHow long the customer has been with you (e.g., months since activation).",
    cols,
    index=cols.index(default_tenure) if default_tenure in cols else 0
)

map_contract = st.sidebar.selectbox(
    "Contract / Plan Type\nThe customer’s commitment type (month-to-month vs 1-year/2-year, etc.).",
    cols,
    index=cols.index(default_contract) if default_contract in cols else 0
)

map_mrc = st.sidebar.selectbox(
    "Monthly Charge (MRC)\nThe customer’s recurring monthly bill amount.",
    cols,
    index=cols.index(default_mrc) if default_mrc in cols else 0
)

map_payment = st.sidebar.selectbox(
    "Payment / Billing Method\nHow the customer pays and receives bills (auto-pay, credit card, paperless billing, etc.).",
    cols,
    index=cols.index(default_payment) if default_payment in cols else 0
)

core_ok = all(x != "(not available)" for x in [map_tenure, map_mrc, map_contract])
if not core_ok:
    st.sidebar.error("Not enough information to score churn reliably. Please upload data with Tenure + Monthly Charge + Contract.")
    st.stop()

# Rename mapped columns into standard names (does not remove other columns)
rename_map = {}
if map_customer != "(not available)":
    rename_map[map_customer] = "customer_id"
rename_map[map_tenure] = "tenure_months"
rename_map[map_contract] = "contract"
rename_map[map_mrc] = "monthly_charge"
rename_map[map_payment] = "paymentmethod"

uploaded_df = uploaded_df.rename(columns=rename_map)

# -----------------------------
# Scoring
# -----------------------------
# If expected_cols couldn’t be inferred, we try to score on whatever is present (may fail).
X_input = align_df_to_model(uploaded_df, expected_cols) if expected_cols else uploaded_df.copy()

try:
    proba = model.predict_proba(X_input)[:, 1]
except Exception as e:
    st.error(f"Scoring failed. Reason: {e}")
    st.stop()

scored_df = uploaded_df.copy()
scored_df["predicted_churn_proba"] = proba
scored_df["risk_band"] = risk_bands_from_proba(proba)
ctx = build_context(scored_df)

st.sidebar.success("Scoring complete ✅")


# -----------------------------
# Main tabs
# -----------------------------
tab_chat, tab_data, tab_how = st.tabs(["Chat assistant", "Data preview", "How it works"])

with tab_data:
    st.subheader("Scored dataset preview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Customers analysed", f"{ctx['n_customers']:,}")
    c2.metric("Avg churn risk", format_pct(ctx["avg_risk"]))
    c3.metric("Top 10% threshold", format_pct(ctx["p90_risk"]))

    st.write("**Risk-band summary**")
    st.dataframe(ctx["segment_summary"], use_container_width=True)

    st.write("**Top 100 customers at risk**")
    top100 = scored_df.sort_values("predicted_churn_proba", ascending=False).head(100)
    st.dataframe(top100, use_container_width=True)

    st.write("**Full scored file (first 25 rows)**")
    st.dataframe(scored_df.head(25), use_container_width=True)

    # Download scored CSV
    csv_bytes = scored_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download scored CSV",
        data=csv_bytes,
        file_name="scored_customers.csv",
        mime="text/csv"
    )

with tab_chat:
    st.subheader("Quick insights (no LLM required)")
    st.caption("This is a simple built-in assistant for a stable demo. You can later swap it with AWS Bedrock.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    q = st.chat_input("Ask: 'How many customers are very high risk?' or 'What should we do next?'")
    if q:
        st.session_state.messages.append({"role": "user", "content": q})

        ql = q.lower()
        if "very high" in ql or "highest" in ql:
            vh = int((scored_df["risk_band"] == "Very high risk").sum())
            reply = f"There are **{vh:,} customers** in the **Very high risk** band. Start with the top 100 list in the Data tab."
        elif "average" in ql or "overall" in ql:
            reply = (
                f"Across **{ctx['n_customers']:,} customers**, the average churn risk is **{format_pct(ctx['avg_risk'])}**. "
                f"The top 10% threshold is **{format_pct(ctx['p90_risk'])}**."
            )
        elif "what should" in ql or "next" in ql or "action" in ql:
            reply = (
                "Suggested next steps:\n"
                "1) Focus outreach on **Very high risk** first (top 100).\n"
                "2) Offer contract upgrades or retention incentives to month-to-month customers.\n"
                "3) Review payment/billing friction (autopay, failed payments, paperless).\n"
                "4) Track churn outcomes over 30 days to validate impact."
            )
        else:
            reply = (
                "Try questions like:\n"
                "- How many customers are very high risk?\n"
                "- What is the overall churn risk?\n"
                "- What should we do next to reduce churn?\n"
                "\nOr use the Data tab to export the scored list."
            )

        with st.chat_message("assistant"):
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})

with tab_how:
    st.subheader("How it works")
    st.markdown(
        """
**Workflow**
1. Upload your trained **.joblib** churn model (built in Jupyter).
2. Upload a customer CSV.
3. Map 5 must-have fields:
   - Customer ID
   - Tenure
   - Contract / Plan Type
   - Monthly Charge (MRC)
   - Payment / Billing Method
4. The app scores churn probability and assigns 4 risk bands:
   - Low / Medium / High / Very high
5. You can download the scored CSV and focus on the top-risk customers.

**Reliability rule**
If Tenure + Monthly Charge + Contract are not available, the app stops with:
Not enough information to score churn reliably.
"""
    )
