import streamlit as st
import pandas as pd

st.set_page_config(page_title="RetentionGPT", layout="wide")

# Load data
segment_df = pd.read_csv("data/segment_summary.csv")
drivers_df = pd.read_csv("data/churn_drivers.csv")
revenue_df = pd.read_csv("data/revenue_at_risk.csv")

st.title("RetentionGPT – AI Retention Assistant")

st.markdown("""
Ask business questions about customer churn, revenue risk, or segments.
This assistant answers using your dashboard data.
""")

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.chat_input("Ask a question")

def generate_response(question):
    q = question.lower()

    if "churn" in q:
        top_driver = drivers_df.sort_values("impact", ascending=False).iloc[0]
        return f"Churn is mainly driven by **{top_driver['driver']}**, contributing the highest impact."

    if "revenue" in q:
        total = revenue_df["revenue_at_risk"].sum()
        return f"Total revenue at risk is approximately **${total:,.0f}**."

    if "segment" in q:
        worst = segment_df.sort_values("churn_rate", ascending=False).iloc[0]
        return f"The highest churn segment is **{worst['segment']}**."

    return "I can answer questions about churn drivers, revenue risk, and customer segments."

if user_input:
    st.session_state.chat.append(("user", user_input))
    answer = generate_response(user_input)
    st.session_state.chat.append(("assistant", answer))

for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(msg)
