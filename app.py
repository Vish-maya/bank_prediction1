import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Bank Term Deposit Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================
# FORCE FULL WIDTH
# =====================================================
st.markdown("""
<style>
.block-container {
    max-width: 100%;
    padding: 2rem;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# THEME & STYLES
# =====================================================
st.markdown("""
<style>
:root {
    --bg: #0b1020;
    --card: #141a2b;
    --text: #e8eaff;
    --muted: #a9b4d1;
    --primary: #6aa3ff;
}
body { background: var(--bg); color: var(--text); }
.card {
    background: linear-gradient(180deg, rgba(20,26,43,.95), rgba(20,26,43,.85));
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,.3);
}
.section-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 12px;
}
.divider {
    height: 1px;
    background: rgba(255,255,255,.1);
    margin: 25px 0;
}
.stButton>button {
    background: linear-gradient(90deg, #6aa3ff, #8fd3f4);
    color: #0b1020;
    font-size: 18px;
    font-weight: 700;
    border-radius: 14px;
    padding: 14px;
    border: none;
}
.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0 10px 25px rgba(106,163,255,.4);
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================
model_path = "model.pkl"
if not os.path.exists(model_path):
    st.error("❌ model.pkl not found")
    st.stop()

model = joblib.load(model_path)
feature_names = model.feature_names_in_

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🧭 Simulation Panel")
st.sidebar.write("Adjust inputs to simulate different customer profiles.")
st.sidebar.markdown("""
**Model Details**
- Algorithm: Random Forest  
- Task: Classification  
- Output: Probability
""")

# =====================================================
# HERO HEADER
# =====================================================
st.markdown("""
<div style="
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    padding: 35px;
    border-radius: 22px;
    margin-bottom: 30px;
    box-shadow: 0 12px 35px rgba(0,0,0,.35);
">
    <h1>🏦 Bank Term Deposit Predictor</h1>
    <p style="color:#dbe4ff;font-size:16px;">
        AI-powered decision support system to identify high-potential customers
        <br>before initiating a marketing campaign.
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# INPUT CARD
# =====================================================
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>📋 Customer Profile</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("👤 Customer Age", 18, 100, 35)
    balance = st.number_input("💰 Account Balance", value=1000.0, step=100.0)
    housing = st.selectbox("🏠 Housing Loan", ["yes", "no"])

with col2:
    loan = st.selectbox("💳 Personal Loan", ["yes", "no"])
    campaign = st.number_input("📞 Campaign Contacts", 1, value=1)
    previous = st.number_input("📂 Previous Contacts", 0, value=0)

st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# PREDICTION BUTTON
# =====================================================
predict = st.button("🔍 Predict Subscription", use_container_width=True)

# =====================================================
# PREDICTION OUTPUT
# =====================================================
if predict:
    X = pd.DataFrame([{
        "age": age,
        "balance": balance,
        "campaign": campaign,
        "previous": previous,
        "housing_yes": 1 if housing == "yes" else 0,
        "loan_yes": 1 if loan == "yes" else 0
    }])

    X = X.reindex(columns=feature_names, fill_value=0)

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    confidence = int(prob * 100)

    colA, colB = st.columns(2)

    # ---------------- RESULT ----------------
    with colA:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📌 Prediction Result")

        if pred == 1:
            st.markdown(
                f"<h2 style='color:#4ade80;'>✅ Likely to Subscribe</h2>"
                f"<p>Confidence: <b>{prob:.2%}</b></p>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<h2 style='color:#f87171;'>❌ Unlikely to Subscribe</h2>"
                f"<p>Confidence: <b>{prob:.2%}</b></p>",
                unsafe_allow_html=True
            )

        st.markdown("#### 📈 Subscription Confidence")
        st.progress(confidence)

        if confidence >= 70:
            st.success("High confidence customer – prioritize outreach")
        elif confidence >= 40:
            st.info("Moderate confidence – targeted follow-up recommended")
        else:
            st.warning("Low confidence – avoid aggressive marketing")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- MODEL INSIGHTS ----------------
    with colB:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("📊 Model Insights")

        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False).head(8)

        fig, ax = plt.subplots()
        ax.barh(fi_df["Feature"], fi_df["Importance"])
        ax.invert_yaxis()
        ax.set_xlabel("Importance Score")
        ax.set_title("Top Influential Features")

        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- AI EXPLANATION ----------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🧠 AI Explanation")

    reasons = []
    if balance > 2000:
        reasons.append("High account balance increases likelihood of subscription")
    if campaign <= 2:
        reasons.append("Lower number of campaign contacts improves success rate")
    if housing == "no":
        reasons.append("No housing loan reduces financial burden")
    if loan == "no":
        reasons.append("No personal loan indicates lower customer risk")

    if reasons:
        for r in reasons:
            st.write("•", r)
    else:
        st.write("Prediction driven by combined customer profile patterns.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- CUSTOMER COMPARISON ----------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📎 Customer Comparison")

    st.write("**Compared to average bank customer:**")
    st.write(f"• Account balance entered: ${balance:,.0f}")
    st.write("• Optimal campaign success occurs within ≤ 2 contacts")
    st.write("• Loan-free profiles show higher subscription probability")

    st.markdown("</div>", unsafe_allow_html=True)

# =====================================================
# FOOTER
# =====================================================
st.markdown(
    "<p style='text-align:center;color:#a9b4d1;font-size:12px;'>"
    "Built with Streamlit • Academic Demonstration"
    "</p>",
    unsafe_allow_html=True
)
