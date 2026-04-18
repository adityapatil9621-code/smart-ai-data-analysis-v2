"""
app.py - Smart AI Data Intelligence System
Additions: Google OAuth, email verification, welcome emails
"""
import random
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from auth_db      import create_tables
from auth_service import (login_user, register_user, verify_email_token,
                          resend_verification, get_username_by_id, upsert_google_user)
from google_oauth  import handle_google_callback, get_google_login_url, is_google_configured
from core_engine   import SmartAIEngine
from suggestion_engine import SuggestionEngine
from chat_engine   import ChatEngine, generate_dynamic_questions

create_tables()
st.set_page_config(page_title="Smart AI Data Intelligence", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#f6fafe}
.glass{background:rgba(255,255,255,0.65);backdrop-filter:blur(16px);border-radius:16px;padding:20px;margin-bottom:20px}
.glass:hover{transform:scale(1.005);transition:.2s ease}
.insight{background:rgba(110,59,216,.05);border-left:3px solid #6e3bd8;padding:10px;border-radius:10px;margin-top:10px}
.auth-card{max-width:440px;margin:40px auto;background:rgba(255,255,255,.85);backdrop-filter:blur(20px);border-radius:20px;padding:36px;box-shadow:0 8px 32px rgba(0,101,146,.1)}
.google-btn{display:flex;align-items:center;justify-content:center;gap:10px;width:100%;padding:12px;border-radius:30px;background:#fff;border:1.5px solid #e8eff4;font-weight:600;font-size:14px;color:#2a3439;cursor:pointer;text-decoration:none;transition:box-shadow .2s;box-shadow:0 2px 8px rgba(0,0,0,.06)}
.google-btn:hover{box-shadow:0 4px 16px rgba(0,0,0,.12)}
.or-div{display:flex;align-items:center;gap:12px;color:#a9b3ba;font-size:12px;margin:18px 0}
.or-div::before,.or-div::after{content:'';flex:1;height:1px;background:#e8eff4}
section[data-testid="stSidebar"]{background:rgba(255,255,255,.6);backdrop-filter:blur(12px)}
.stButton>button{border-radius:20px;font-weight:600}
.verify-banner{background:linear-gradient(135deg,#006592,#005980);color:#f5f9ff;border-radius:14px;padding:16px 20px;margin-bottom:16px;font-size:14px}
.ok-banner{background:linear-gradient(135deg,#006d4a,#005f40);color:#f5f9ff;border-radius:14px;padding:20px 24px;margin-bottom:20px}
.ok-banner h3{margin:0 0 6px;font-family:'Space Grotesk',sans-serif}
.ok-banner p{margin:0;font-size:13px;opacity:.85}
</style>""", unsafe_allow_html=True)

# Session state defaults
for k,v in [("page","login"),("user_id",None),("just_verified",False),("resend_mode",False)]:
    if k not in st.session_state: st.session_state[k]=v

# Google OAuth callback
google_user = handle_google_callback()
if google_user:
    uid = upsert_google_user(google_user["sub"], google_user["email"], google_user["name"])
    if uid:
        st.session_state.user_id=uid; st.session_state.page="home"; st.rerun()
    else:
        st.error("Google sign-in failed. Please try again.")

# Email verification token in URL
params = st.query_params
if "verify_token" in params:
    token = params["verify_token"]
    st.query_params.clear()
    ok, reason = verify_email_token(token)
    st.session_state.just_verified = ok
    if not ok: st.error(f"Verification failed: {reason}")
    st.session_state.page = "login"

GOOGLE_SVG = ('<svg width="18" height="18" viewBox="0 0 18 18" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844a4.14 4.14 0 0 1-1.796 2.716v2.259h2.908c1.702-1.567 2.684-3.875 2.684-6.615z" fill="#4285F4"/>'
    '<path d="M9 18c2.43 0 4.467-.806 5.956-2.184l-2.908-2.259c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 0 0 9 18z" fill="#34A853"/>'
    '<path d="M3.964 10.706A5.41 5.41 0 0 1 3.682 9c0-.593.102-1.17.282-1.706V4.962H.957A8.996 8.996 0 0 0 0 9c0 1.452.348 2.827.957 4.038l3.007-2.332z" fill="#FBBC05"/>'
    '<path d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 0 0 .957 4.962L3.964 7.294C4.672 5.163 6.656 3.58 9 3.58z" fill="#EA4335"/></svg>')

def google_btn(label="Continue with Google"):
    url = get_google_login_url()
    if not url: return
    st.markdown(f'<a class="google-btn" href="{url}">{GOOGLE_SVG} {label}</a>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ── LOGIN PAGE ────────────────────────────────────────────────
if st.session_state.page == "login":
    if st.session_state.just_verified:
        st.markdown('<div class="ok-banner"><h3>🎉 Email verified!</h3><p>Your account is active. Log in below.</p></div>', unsafe_allow_html=True)
        st.session_state.just_verified = False

    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    st.markdown('<h2 style="font-family:Space Grotesk,sans-serif;font-size:26px;font-weight:700;margin-bottom:20px">🔐 Sign In</h2>', unsafe_allow_html=True)

    google_btn()
    if is_google_configured():
        st.markdown('<div class="or-div">or</div>', unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        sub = st.form_submit_button("Sign In", use_container_width=True)

    if sub:
        uid, reason = login_user(username, password)
        if uid:
            st.session_state.user_id=uid; st.session_state.page="home"; st.rerun()
        else:
            st.error(reason)
            if "verify" in reason.lower(): st.session_state.resend_mode = True

    if st.session_state.get("resend_mode"):
        st.markdown("---")
        st.markdown("**Didn't receive the verification email?**")
        re_email = st.text_input("Your email address:", key="resend_inp")
        if st.button("📨 Resend Verification Email"):
            if re_email:
                ok, msg = resend_verification(re_email)
                (st.success("Sent! Check your inbox.") or st.session_state.update(resend_mode=False)) if ok else st.error(msg)
            else:
                st.warning("Please enter your email.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.write("Don't have an account?")
    if st.button("Create Account →"):
        st.session_state.page="register"; st.session_state.resend_mode=False; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ── REGISTER PAGE ─────────────────────────────────────────────
elif st.session_state.page == "register":
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    st.markdown('<h2 style="font-family:Space Grotesk,sans-serif;font-size:26px;font-weight:700;margin-bottom:20px">📝 Create Account</h2>', unsafe_allow_html=True)

    google_btn("Sign up with Google")
    if is_google_configured():
        st.markdown('<div class="or-div">or sign up with email</div>', unsafe_allow_html=True)

    with st.form("register_form"):
        username = st.text_input("Username")
        email    = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm  = st.text_input("Confirm Password", type="password")
        sub      = st.form_submit_button("Create Account", use_container_width=True)

    if sub:
        if password != confirm:
            st.error("Passwords do not match.")
        else:
            ok, reason = register_user(username, email, password)
            if ok:
                st.markdown(f'<div class="verify-banner">📧 <strong>Check your inbox!</strong><br>We sent a verification link to <strong>{email}</strong>. Click it to activate your account before logging in.</div>', unsafe_allow_html=True)
            else:
                st.error(reason)

    if st.button("← Back to Login"):
        st.session_state.page="login"; st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ── HOME PAGE ─────────────────────────────────────────────────
elif st.session_state.page == "home":
    if not st.session_state.user_id:
        st.session_state.page="login"; st.rerun()

    username = get_username_by_id(st.session_state.user_id)

    with st.sidebar:
        st.markdown(f"### ⚙️ Settings")
        st.write(f"Logged in as: **{username}**")
        view_mode = st.radio("View Mode", ["Summary", "Detailed"])
        st.markdown("---")
        if st.button("🚪 Logout"):
            st.session_state.user_id=None; st.session_state.page="login"; st.rerun()

    st.title("🧠 Smart AI Data Intelligence")
    st.markdown("Upload a CSV dataset and let the AI analyse it for you.")
    st.markdown("---")

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📂 Upload Dataset (CSV)", type=["csv"])
    st.markdown('</div>', unsafe_allow_html=True)

    if not uploaded_file:
        st.info("Please upload a CSV file to get started."); st.stop()

    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        st.error("Could not parse the file."); st.stop()

    if df.empty or len(df) < 10:
        st.error("Dataset too small (minimum 10 rows)."); st.stop()

    st.success(f"✅ Loaded: {len(df):,} rows × {len(df.columns)} columns")
    st.dataframe(df.head(5), use_container_width=True)

    if st.button("🔍 Run Analysis", type="primary"):
        with st.spinner("Running AI pipeline…"):
            try:
                st.session_state.memory = SmartAIEngine().run_pipeline(df)
            except Exception as e:
                st.error(f"Analysis failed: {e}"); st.stop()

    if "memory" not in st.session_state: st.stop()

    memory=st.session_state.memory; model_info=memory["model_intelligence"]
    insight=memory["insight_intelligence"]; forecast=memory.get("forecast_intelligence")
    score=memory["intelligence_score"]; meta=memory["metadata"]
    strategic_obj = SuggestionEngine().run(insight_obj=insight, forecast_obj=forecast, score_obj=score)

    st.success("✅ Analysis complete!")

    report = f"SMART AI REPORT\nUser: {username}\nRows: {meta['rows']} | Cols: {meta['columns']} | Quality: {meta['quality_score']}\nDomain: {meta.get('domain','N/A')}\nModel: {model_info['selected_model']}\nScore: {score['score']} | Grade: {score['grade']}"
    st.download_button("📥 Download Report", report, file_name="smart_ai_report.txt")

    if meta["quality_score"]<0.6: st.warning("⚠️ Low data quality detected.")
    if score["score"]<0.5: st.warning("⚠️ Low intelligence score.")

    if view_mode == "Summary":
        st.header("📊 Executive Dashboard")
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(f'<div class="glass"><h4>Rows</h4><h2>{meta["rows"]:,}</h2></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="glass"><h4>Quality</h4><h2>{meta["quality_score"]:.0%}</h2></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="glass"><h4>Score</h4><h2>{score["score"]:.0%}</h2></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="glass"><h4>Grade</h4><h2>{score["grade"]}</h2></div>', unsafe_allow_html=True)
        st.success(f"**Model:** {model_info['selected_model']}")
        if insight["top_positive_drivers"]:
            t=insight["top_positive_drivers"][0]; st.info(f"🔑 Top Driver: **{t['feature']}** (impact: {t['impact']})")
        visuals=memory["visual_intelligence"]["figures"]; shown=0
        for key in ["time_series","heatmap","distribution","boxplot","relationship"]:
            if key in visuals and shown<2:
                st.markdown('<div class="glass">', unsafe_allow_html=True)
                st.pyplot(visuals[key], use_container_width=False)
                st.markdown('</div>', unsafe_allow_html=True); shown+=1
    else:
        st.header("🔬 Detailed Analysis")
        col1,col2,col3=st.columns(3)
        col1.metric("Model",model_info["selected_model"]); col2.metric("Confidence",f"{model_info['confidence']:.3f}"); col3.metric("Stability",f"{model_info['stability']:.3f}")
        cp,cn=st.columns(2)
        with cp:
            st.markdown('<div class="glass">', unsafe_allow_html=True); st.subheader("✅ Positive Drivers")
            [st.markdown(f"• **{d['feature']}** → {d['impact']}") for d in insight["top_positive_drivers"]]
            st.markdown('</div>', unsafe_allow_html=True)
        with cn:
            st.markdown('<div class="glass">', unsafe_allow_html=True); st.subheader("⚠️ Negative Drivers")
            [st.markdown(f"• **{d['feature']}** → {d['impact']}") for d in insight["top_negative_drivers"]] if insight["top_negative_drivers"] else st.write("None identified.")
            st.markdown('</div>', unsafe_allow_html=True)
        if insight.get("feature_importance_plot"): st.subheader("📊 Feature Importance"); st.pyplot(insight["feature_importance_plot"], use_container_width=False)
        if insight.get("shap_plot"): st.subheader("🧠 SHAP"); st.pyplot(insight["shap_plot"], use_container_width=False)
        st.subheader("📈 Visuals")
        [st.pyplot(memory["visual_intelligence"]["figures"][k], use_container_width=False) for k in ["time_series","heatmap","distribution","boxplot","relationship"] if k in memory["visual_intelligence"]["figures"]]
        st.markdown('<div class="glass">', unsafe_allow_html=True); st.subheader("🔮 Forecast")
        if forecast and forecast.get("forecast_values"):
            st.info(f"Target: **{forecast.get('target_column','?')}**")
            fc1,fc2,fc3=st.columns(3)
            fc1.metric("Trend",forecast.get("trend_direction","N/A")); fc2.metric("Volatility",f"{forecast.get('volatility_score',0):.2f}"); fc3.metric("Confidence",f"{forecast.get('forecast_confidence',0):.2f}")
            st.line_chart(pd.Series(forecast["forecast_values"]),height=250)
        else:
            st.info("No time-series detected.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.subheader("🎯 Recommendations")
        r1,r2,r3=st.columns(3)
        with r1: st.markdown("**📈 Growth**"); [st.write(f"• {g}") for g in strategic_obj.growth_opportunities]
        with r2: st.markdown("**⚠️ Risk**"); [st.write(f"• {r}") for r in strategic_obj.risk_mitigation_actions]
        with r3: st.markdown("**🛠️ Stability**"); [st.write(f"• {s}") for s in strategic_obj.stability_recommendations]
        st.caption(f"*{strategic_obj.human_oversight_note}*")

    st.markdown("---"); st.markdown('<div class="glass">', unsafe_allow_html=True); st.header("💬 AI Chat")
    if st.button("🤖 Generate Questions"):
        with st.spinner("Generating…"): st.session_state.suggested_questions = generate_dynamic_questions(memory)
    qs = st.session_state.get("suggested_questions",[])
    if qs:
        st.subheader("💡 Suggested"); cols=st.columns(2)
        for i,q in enumerate(random.sample(qs, min(6,len(qs)))):
            if cols[i%2].button(q, key=f"q_{i}"): st.session_state.chat_prefill=q
    prefill=st.session_state.pop("chat_prefill","")
    uq=st.text_input("Your question:", value=prefill, key="chat_field")
    st.markdown('</div>', unsafe_allow_html=True)
    if uq:
        with st.spinner("Thinking…"): st.success(ChatEngine().respond(uq, memory))
    with st.expander("🛠️ Debug"):
        st.json({k:v for k,v in memory.items() if k not in ("cleaned_df","visual_intelligence")})