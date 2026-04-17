"""
app.py

Streamlit Interface for Smart AI Data Intelligence System.

Improvements:
- Sidebar-based navigation (cleaner layout)
- Validation before running pipeline
- Error shown cleanly via st.error, not unhandled exceptions
- Download report includes domain + grade
- Suggested questions shown in a grid (2 cols) for readability
- Forecast target column labelled correctly
- View-mode radio moved to sidebar
"""

import random
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from auth_db      import create_tables
from auth_service import login_user, register_user
from core_engine  import SmartAIEngine
from suggestion_engine import SuggestionEngine
from chat_engine  import ChatEngine, generate_dynamic_questions


# ===============================
# Init
# ===============================
create_tables()

st.set_page_config(page_title="Smart AI Data Intelligence", layout="wide")
st.markdown("""
<style>

/* ===== Fonts ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&family=Space+Grotesk:wght@500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #f6fafe;
}

/* ===== Title ===== */
.title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 42px;
    font-weight: 700;
    letter-spacing: -1px;
}

/* ===== Glass Card ===== */
.glass {
    background: rgba(255,255,255,0.6);
    backdrop-filter: blur(16px);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 20px;
}

/* ===== Insight Box ===== */
.insight {
    background: rgba(110,59,216,0.05);
    border-left: 3px solid #6e3bd8;
    padding: 10px;
    border-radius: 10px;
    margin-top: 10px;
}

/* ===== Sidebar Styling ===== */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.6);
    backdrop-filter: blur(12px);
}

/* ===== Button Upgrade ===== */
.stButton>button {
    border-radius: 20px;
    font-weight: 600;
}
.glass:hover {
    transform: scale(1.01);
    transition: 0.2s ease;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# Session State Defaults
# ===============================
for key, default in [("page", "login"), ("user_id", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ===============================
# PAGE: LOGIN
# ===============================
if st.session_state.page == "login":
    st.title("🔐 Smart AI — Login")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        uid = login_user(username, password)
        if uid:
            st.session_state.user_id = uid
            st.session_state.page    = "home"
            st.rerun()
        else:
            st.error("Invalid username or password.")

    st.write("Don't have an account?")
    if st.button("Register"):
        st.session_state.page = "register"
        st.rerun()


# ===============================
# PAGE: REGISTER
# ===============================
elif st.session_state.page == "register":
    st.title("📝 Create Account")

    with st.form("register_form"):
        username = st.text_input("Username")
        email    = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Create Account")

    if submitted:
        if len(password) < 6:
            st.error("Password must be at least 6 characters.")
        elif register_user(username, email, password):
            st.success("Account created! Please login.")
            st.session_state.page = "login"
            st.rerun()
        else:
            st.error("Username or email already in use.")

    if st.button("← Back to Login"):
        st.session_state.page = "login"
        st.rerun()


# ===============================
# PAGE: HOME
# ===============================
elif st.session_state.page == "home":

    if st.session_state.user_id is None:
        st.session_state.page = "login"
        st.rerun()

    # ---- Sidebar ----
    with st.sidebar:
        st.title("⚙️ Settings")
        st.write(f"Logged in as: **{st.session_state.user_id}**")
        view_mode = st.radio("View Mode", ["Summary", "Detailed"])
        if st.button("🚪 Logout"):
            st.session_state.user_id = None
            st.session_state.page    = "login"
            st.rerun()

    # ---- Header ----
    st.title("🧠 Smart AI Data Intelligence")
    st.markdown("Upload a CSV dataset and let the AI analyse it for you.")
    st.markdown("---")

    # ---- File Upload ----
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📂 Upload Dataset (CSV)", type=["csv"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.info("Please upload a CSV file to get started.")
        st.stop()

    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        st.error("Could not parse the file. Please upload a valid CSV.")
        st.stop()

    if df.empty or len(df) < 10:
        st.error("Dataset is too small (minimum 10 rows). Please upload a larger file.")
        st.stop()

    st.success(f"✅ File loaded: {len(df):,} rows × {len(df.columns)} columns")
    st.dataframe(df.head(5), use_container_width=True)

    # ---- Run Analysis ----
    if st.button("🔍 Run Analysis", type="primary"):
        with st.spinner("Running AI pipeline…"):
            try:
                engine = SmartAIEngine()
                st.session_state.memory = engine.run_pipeline(df)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

    if "memory" not in st.session_state:
        st.stop()

    # ================================================================
    # RESULTS
    # ================================================================
    memory       = st.session_state.memory
    model_info   = memory["model_intelligence"]
    insight      = memory["insight_intelligence"]
    forecast     = memory.get("forecast_intelligence")
    score        = memory["intelligence_score"]
    meta         = memory["metadata"]

    suggestion_engine = SuggestionEngine()
    strategic_obj = suggestion_engine.run(
        insight_obj=insight,
        forecast_obj=forecast,
        score_obj=score,
    )

    st.success("✅ Analysis complete!")

    # ---- Download Report ----
    report_text = f"""
SMART AI INTELLIGENCE REPORT
==============================
Rows:    {meta['rows']}
Columns: {meta['columns']}
Quality: {meta['quality_score']}
Domain:  {meta.get('domain', 'N/A')}

MODEL
-----
Selected: {model_info['selected_model']}
Confidence: {model_info['confidence']}
Stability:  {model_info['stability']}

INTELLIGENCE SCORE
------------------
Score:      {score['score']}
Grade:      {score['grade']}
Confidence: {score['confidence_level']}

TOP POSITIVE DRIVERS
--------------------
{chr(10).join(f"  • {d['feature']}  (impact={d['impact']})" for d in insight['top_positive_drivers'])}

STRATEGIC RECOMMENDATIONS
--------------------------
Growth:
{chr(10).join(f"  📈 {g}" for g in strategic_obj.growth_opportunities)}

Risk:
{chr(10).join(f"  ⚠️ {r}" for r in strategic_obj.risk_mitigation_actions)}

Stability:
{chr(10).join(f"  🛠️ {s}" for s in strategic_obj.stability_recommendations)}

{strategic_obj.human_oversight_note}
"""
    st.download_button("📥 Download Report", report_text, file_name="smart_ai_report.txt")

    if meta["quality_score"] < 0.6:
        st.warning("⚠️ Low data quality detected — results may be less reliable.")
    if score["score"] < 0.5:
        st.warning("⚠️ Low intelligence score — consider enriching your dataset.")

    # ==============================================================
    # SUMMARY MODE
    # ==============================================================
    if view_mode == "Summary":
        st.header("📊 Executive Dashboard")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(f'<div class="glass"><h4>Rows</h4><h2>{meta["rows"]:,}</h2></div>', unsafe_allow_html=True)

        with c2:
            st.markdown(f'<div class="glass"><h4>Quality</h4><h2>{meta["quality_score"]:.0%}</h2></div>',
                        unsafe_allow_html=True)

        with c3:
            st.markdown(f'<div class="glass"><h4>Score</h4><h2>{score["score"]:.0%}</h2></div>', unsafe_allow_html=True)

        with c4:
            st.markdown(f'<div class="glass"><h4>Grade</h4><h2>{score["grade"]}</h2></div>', unsafe_allow_html=True)

        st.success(f"**Model:** {model_info['selected_model']}")

        if insight["top_positive_drivers"]:
            top = insight["top_positive_drivers"][0]
            st.info(f"🔑 Top Driver: **{top['feature']}** (impact: {top['impact']})")

        visuals = memory["visual_intelligence"]["figures"]
        order   = ["time_series", "heatmap", "distribution", "boxplot", "relationship"]
        shown   = 0
        for key in order:
            if key in visuals and shown < 2:
                st.markdown('<div class="glass">', unsafe_allow_html=True)
                st.pyplot(visuals[key], use_container_width=False)
                st.markdown('</div>', unsafe_allow_html=True)
                shown += 1

    # ==============================================================
    # DETAILED MODE
    # ==============================================================
    else:
        st.header("🔬 Detailed Analysis")

        # Model info
        st.subheader("📌 Model Information")
        col1, col2, col3 = st.columns(3)
        col1.metric("Model",      model_info["selected_model"])
        col2.metric("Confidence", f"{model_info['confidence']:.3f}")
        col3.metric("Stability",  f"{model_info['stability']:.3f}")

        # Drivers
        col_pos, col_neg = st.columns(2)

        with col_pos:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.subheader("✅ Positive Drivers")
            for d in insight["top_positive_drivers"]:
                st.markdown(f"• **{d['feature']}** → {d['impact']}")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_neg:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.subheader("⚠️ Negative Drivers")
            if insight["top_negative_drivers"]:
                for d in insight["top_negative_drivers"]:
                    st.markdown(f"• **{d['feature']}** → {d['impact']}")
            else:
                st.write("No negative drivers")
            st.markdown('</div>', unsafe_allow_html=True)

        # Feature importance
        if insight.get("feature_importance_plot"):
            st.subheader("📊 Feature Importance")
            st.pyplot(insight["feature_importance_plot"], use_container_width=False)

        # SHAP
        if insight.get("shap_plot"):
            st.subheader("🧠 SHAP Explainability")
            st.pyplot(insight["shap_plot"], use_container_width=False)

        # Visuals
        st.subheader("📈 Visual Intelligence")
        visuals = memory["visual_intelligence"]["figures"]
        for key in ["time_series", "heatmap", "distribution", "boxplot", "relationship"]:
            if key in visuals:
                st.pyplot(visuals[key], use_container_width=False)

        # Forecast
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("🔮 Forecast")
        if forecast and forecast.get("forecast_values"):
            target_label = forecast.get("target_column", "Target")
            st.info(f"Forecast target: **{target_label}**")

            fc1, fc2, fc3 = st.columns(3)
            fc1.metric("Trend",      forecast.get("trend_direction", "N/A"))
            fc2.metric("Volatility", f"{forecast.get('volatility_score', 0):.2f}")
            fc3.metric("Confidence", f"{forecast.get('forecast_confidence', 0):.2f}")

            forecast_series = pd.Series(forecast["forecast_values"])
            st.line_chart(forecast_series, height=250)

            if "confidence_band" in forecast:
                lower = forecast["confidence_band"]["lower"]
                upper = forecast["confidence_band"]["upper"]
                fig_f, ax_f = plt.subplots(figsize=(7, 3))
                x = list(range(len(lower)))
                ax_f.plot(x, forecast["forecast_values"], label="Forecast", linewidth=2)
                ax_f.fill_between(x, lower, upper, alpha=0.2, label="90% Confidence Band")
                ax_f.set_title("Forecast with Confidence Interval")
                ax_f.legend()
                plt.tight_layout()
                st.pyplot(fig_f, use_container_width=False)
        else:
            st.info("Forecast not available — no time-series structure detected in this dataset.")
        st.markdown('</div>', unsafe_allow_html=True)

        # Recommendations
        st.subheader("🎯 Strategic Recommendations")
        rec1, rec2, rec3 = st.columns(3)
        with rec1:
            st.markdown("**📈 Growth Opportunities**")
            for g in strategic_obj.growth_opportunities:
                st.write(f"• {g}")
        with rec2:
            st.markdown("**⚠️ Risk Mitigation**")
            for r in strategic_obj.risk_mitigation_actions:
                st.write(f"• {r}")
        with rec3:
            st.markdown("**🛠️ Stability**")
            for s in strategic_obj.stability_recommendations:
                st.write(f"• {s}")

        st.caption(f"*{strategic_obj.human_oversight_note}*")

    # ==============================================================
    # CHAT
    # ==============================================================
    st.markdown("---")
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.header("💬 AI Chat Assistant")

    if st.button("🤖 Generate Smart Questions"):
        with st.spinner("Generating questions…"):
            st.session_state.suggested_questions = generate_dynamic_questions(memory)

    questions = st.session_state.get("suggested_questions", [])
    if questions:
        st.subheader("💡 Suggested Questions")
        sample = random.sample(questions, min(6, len(questions)))
        cols   = st.columns(2)
        for i, q in enumerate(sample):
            if cols[i % 2].button(q, key=f"q_{i}"):
                st.session_state.chat_prefill = q

    chat_engine = ChatEngine()
    prefill      = st.session_state.pop("chat_prefill", "")
    user_query   = st.text_input("Enter your question:", value=prefill, key="chat_input_field")
    st.markdown('</div>', unsafe_allow_html=True)

    if user_query:
        with st.spinner("Thinking…"):
            response = chat_engine.respond(user_query, memory)
        st.success(response)

    # Debug expander
    with st.expander("🛠️ Debug — System Memory"):
        safe_memory = {k: v for k, v in memory.items() if k not in ("cleaned_df", "visual_intelligence")}
        st.json(safe_memory)