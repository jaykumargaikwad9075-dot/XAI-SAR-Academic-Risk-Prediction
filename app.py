"""XAI-SAR: An Explainable AI System for Academic Risk Prediction and
Personalised Student Recommendations.  Run:  streamlit run app.py"""
import os, sys, json, time, smtplib, ssl, joblib, numpy as np, pandas as pd, shap, streamlit as st
from email.mime.text import MIMEText
import plotly.graph_objects as go
import plotly.express as px
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from xai_core import *

# ============================================================================ PAGE & THEME
st.set_page_config(page_title="XAI-SAR | Student Success Intelligence", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

# Palette matched to the reference dashboard: near-black sidebar/buttons, sage-off-white
# page background, white cards, a single lime accent for highlights.
INK, MUTE = "#17233D", "#66748B"
BLACK = "#17233D"
BG = "#F3F6FC"
CARD = "#FFFFFF"
BORDER = "#E2E8F2"
LIME = "#27D7B0"
LIME_TEXT = "#083C39"
COL = {"Low Risk": "#10A879", "Medium Risk": "#F59E0B", "High Risk": "#E5485D"}
COL_SOFT = {"Low Risk": "#E1F8EF", "Medium Risk": "#FFF4D8", "High Risk": "#FFE7EB"}
PLOT_FONT = dict(family="Inter, sans-serif", color=INK, size=12)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
:root {{ color-scheme: light only; }}
html, body, [class*="css"] {{ font-family:'DM Sans',sans-serif; }}
h1,h2,h3,h4,h5 {{ font-family:'Manrope','DM Sans',sans-serif; }}
.stApp {{ background:linear-gradient(135deg,#F5F7FC 0%,#EEF3FA 100%) !important; }}
[data-testid="stHeader"] {{ background:transparent !important; }}
[data-testid="stMain"] {{ color:{INK} !important; }}
[data-testid="stMain"] label,[data-testid="stMain"] p,[data-testid="stMain"] li,[data-testid="stMain"] span {{ color:inherit; }}
[data-testid="stMain"] [data-testid="stWidgetLabel"] p {{ color:{INK} !important;font-weight:600; }}
[data-testid="stMain"] [data-baseweb="select"] * {{ color:{INK} !important; }}
[data-testid="stMain"] [data-baseweb="input"] input {{ color:{INK} !important;background:#fff !important;border-radius:10px !important; }}
[data-testid="stMain"] [data-baseweb="tag"] {{ background:{BLACK} !important; }}
[data-testid="stMain"] [data-baseweb="tag"] span {{ color:white !important; }}
[data-testid="stMain"] [data-testid="stCaptionContainer"] {{ color:{MUTE} !important; }}
[data-testid="stMain"] [data-testid="stDataFrame"] {{ background:#fff !important;border:1px solid {BORDER};border-radius:14px;overflow:hidden; }}
[data-testid="stMain"] [data-testid="stMetricValue"] {{ color:{INK} !important; }}
section[data-testid="stSidebar"] {{ background:linear-gradient(180deg,#111B32 0%,#1B2B4A 100%) !important;border-right:0; }}
section[data-testid="stSidebar"] * {{ color:#F5F8FF !important; }}
section[data-testid="stSidebar"] input,section[data-testid="stSidebar"] textarea {{ color:{INK} !important; }}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {{ font-size:1.15rem;font-weight:800;letter-spacing:-.3px; }}
section[data-testid="stSidebar"] [data-baseweb="radio"] div[role="radio"] {{ border-color:#B6C5DF !important; }}
/* navigation */
div[data-baseweb="tab-list"] {{ gap:8px;background:#E7ECF5;padding:7px;border-radius:14px;border:1px solid #DFE6F1;width:fit-content; }}
button[data-baseweb="tab"] {{ background:transparent !important;border-radius:10px !important;padding:10px 18px !important;font-weight:700 !important;color:#65728A !important;border:0 !important; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color:white !important;background:{BLACK} !important;box-shadow:0 4px 12px rgba(23,35,61,.18); }}
div[data-baseweb="tab-highlight"],div[data-baseweb="tab-border"] {{ display:none !important; }}
.hero {{ display:flex;align-items:center;justify-content:space-between;margin:8px 0 26px;padding:26px 28px;flex-wrap:wrap;gap:16px;background:linear-gradient(120deg,#17233D 0%,#263E67 70%,#285B72 100%);border-radius:22px;box-shadow:0 12px 30px rgba(28,47,82,.15); }}
.hero h1 {{ color:#fff;font-size:2rem;margin:0;font-weight:800;letter-spacing:-.8px; }}
.hero h1 span {{ color:#9DEAD9 !important; }}
.hero div div {{ color:#C5D2E7 !important; }}
.badge {{ display:inline-block;background:rgba(255,255,255,.1);color:#F8FAFF !important;padding:8px 13px;border-radius:999px;font-size:.74rem;font-weight:700;margin:4px 0 4px 6px;border:1px solid rgba(255,255,255,.2);backdrop-filter:blur(8px); }}
.badge.lime {{ background:#27D7B0;color:#073B38 !important;border-color:#27D7B0; }}
.kpi {{ background:#fff;border-radius:18px;padding:19px 20px;box-shadow:0 6px 18px rgba(28,47,82,.055);border:1px solid #E5EAF3;height:100%;position:relative;overflow:hidden; }}
.kpi:before {{ content:'';position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--kc,#27D7B0); }}
.kpi .top {{ display:flex;align-items:center;justify-content:space-between;margin-bottom:12px; }}
.kpi .icon {{ width:36px;height:36px;border-radius:12px;background:var(--kc,#27D7B0)22;display:flex;align-items:center;justify-content:center;font-size:1rem; }}
.kpi .num {{ font-size:1.85rem;font-weight:800;color:{INK};line-height:1.1;letter-spacing:-.7px; }}
.kpi .lab {{ font-size:.8rem;color:{MUTE};margin-top:7px;font-weight:600; }}
.kpi .delta {{ font-size:.72rem;margin-top:7px;font-weight:800; }}
.section-title {{ background:#fff;border:1px solid {BORDER};border-left:4px solid #27D7B0;border-radius:14px;padding:14px 17px;margin:8px 0 10px;box-shadow:0 4px 12px rgba(28,47,82,.035);font-size:.82rem;font-weight:800;letter-spacing:.65px;text-transform:uppercase;color:{INK}; }}
[data-testid="stPlotlyChart"] {{ background:#fff;border:1px solid {BORDER};border-radius:16px;padding:8px 8px 2px;box-shadow:0 5px 16px rgba(28,47,82,.045); }}
.stDataFrame {{ border-radius:14px; }}
.risk-pill {{ display:inline-block;padding:7px 20px;border-radius:999px;font-weight:700;font-size:1rem; }}
.metric-box {{ background:#F3F6FC;border-radius:12px;padding:12px 14px;text-align:center;border:1px solid {BORDER}; }}
.metric-box .num {{ font-size:1.15rem;font-weight:800;color:{INK}; }}
.metric-box .lab {{ font-size:.68rem;color:{MUTE};text-transform:uppercase;letter-spacing:.5px;margin-top:2px; }}
.rec {{ background:#F3F6FC;border-left:4px solid #27D7B0;border-radius:10px;padding:12px 15px;margin-bottom:9px;font-size:.88rem;color:{INK}; }}
.driver-chip {{ display:inline-block;background:#FFE7EB;color:#A52D43;padding:5px 12px;border-radius:999px;font-size:.76rem;margin:3px;font-weight:700;border:1px solid #F5C5CF; }}
.protect-chip {{ display:inline-block;background:#E1F8EF;color:#087451;padding:5px 12px;border-radius:999px;font-size:.76rem;margin:3px;font-weight:700;border:1px solid #BDEBD8; }}
.tag {{ display:inline-block;background:#EAF0F8;color:{INK};padding:3px 10px;border-radius:999px;font-size:.72rem;font-weight:700;margin-left:8px;border:1px solid {BORDER}; }}
.email-box {{ background:#F3F6FC;border:1px solid {BORDER};border-radius:12px;padding:16px 18px;font-family:monospace;font-size:.83rem;white-space:pre-wrap;color:{INK};line-height:1.5; }}
hr {{ border-color:{BORDER}; }}
footer,#MainMenu,[data-testid="stMainMenu"] {{ visibility:hidden; }}
.stButton button,.stDownloadButton button {{ background:{BLACK} !important;color:#fff !important;border-radius:11px !important;border:0 !important;font-weight:700 !important;padding:9px 19px !important;transition:all .18s ease; }}
.stButton button:hover,.stDownloadButton button:hover {{ background:#285B72 !important;box-shadow:0 5px 14px rgba(23,35,61,.2);transform:translateY(-1px); }}
.stButton button:disabled {{ background:#C9D2E1 !important;color:#78859A !important; }}
</style>
""", unsafe_allow_html=True)


# ============================================================================ DATA / MODEL
@st.cache_resource
def load():
    bundle = joblib.load("outputs/risk_model.joblib")
    explainer = shap.TreeExplainer(bundle["model"])
    data = pd.read_csv("data/student_dataset_201.csv")
    metrics = json.load(open("outputs/metrics.json"))
    try:
        extra = pd.read_csv("data/additional_students.csv")
    except FileNotFoundError:
        extra = pd.DataFrame(columns=data.columns)
    Xall = build_features(data)
    sv_all = shap_risk_score(explainer, Xall)
    proba_all = bundle["model"].predict_proba(Xall)
    preds_all = pd.Series(np.array(CLASSES)[proba_all.argmax(1)], index=data.index)
    return bundle["model"], explainer, data, extra, bundle["name"], metrics, Xall, sv_all, proba_all, preds_all

try:
    model, explainer, data, extra, model_name, metrics, Xall, sv_all, proba_all, preds_all = load()
except FileNotFoundError as e:
    st.error(f"Could not find a required file: `{e.filename}`.\n\nRun `python src/train_and_explain.py` from the "
              "`xai_project` folder first so the `outputs/` and `data/` files exist, then restart this app.")
    st.stop()

N = len(data)
data_ext = data.copy()
data_ext["Predicted Risk"] = preds_all
data_ext["P(High Risk)"] = proba_all[:, CLASS_ID["High Risk"]].round(3)

# "Directory" = the 201-student research cohort + any individually-added contacts (e.g. STU201).
# Cohort-wide statistics (Overview / Model Performance) use `data` / `data_ext` only, so they stay
# consistent with the Part I report; the directory is only used for picking/emailing individuals.
if len(extra):
    Xe = build_features(extra)
    pe = model.predict_proba(Xe)
    extra_ext = extra.copy()
    extra_ext["Predicted Risk"] = np.array(CLASSES)[pe.argmax(1)]
    extra_ext["P(High Risk)"] = pe[:, CLASS_ID["High Risk"]].round(3)
    directory = pd.concat([data_ext, extra_ext], ignore_index=True)
else:
    directory = data_ext

def risk_pill(risk, size="1.0rem"):
    return f"<span class='risk-pill' style='background:{COL_SOFT[risk]}; color:{COL[risk]}; font-size:{size};'>● {risk}</span>"

def kpi(col, num, lab, color, delta=None, icon="●"):
    d = f"<div class='delta' style='color:{color}'>{delta}</div>" if delta else ""
    col.markdown(f"<div class='kpi'><div class='top'><span class='icon' style='--kc:{color}22'>{icon}</span></div>"
                 f"<div class='num'>{num}</div><div class='lab'>{lab}</div>{d}</div>", unsafe_allow_html=True)

def get_report(row):
    feats = build_features(row.to_frame().T.astype({c: float for c in MARK_COLS + ["Attendance (%)", "Study Hours (per day)"]})).iloc[0]
    return feats, explain_student(model, explainer, row, feats)

# ============================================================================ HEADER
st.markdown(f"""
<div class="hero">
  <div><h1>XAI-SAR <span style="font-size:.55em; font-weight:600; color:#6B7280;">STUDENT SUCCESS INTELLIGENCE</span></h1><div style="color:#6B7280; font-size:.82rem; margin-top:5px;">ACADEMIC INTELLIGENCE  /  EXPLAINABLE PREDICTIONS  /  PERSONALIZED SUPPORT</div></div>
  <div>
    <span class="badge lime">{N} students monitored</span>
    <span class="badge">Model: {model_name}</span>
    <span class="badge">{metrics['comparison'][2]['CV Accuracy']:.0%} cross-val accuracy</span>
  </div>
</div>
""", unsafe_allow_html=True)

tab_overview, tab_explorer, tab_email, tab_model, tab_about = st.tabs(
    ["Overview", "Student Explorer", "Email Center", "Model Performance", "About"])

# ============================================================================ TAB 1 — OVERVIEW
with tab_overview:
    vc = data["Academic Risk"].value_counts().reindex(CLASSES)

    k1, k2, k3, k4, k5 = st.columns(5)
    kpi(k1, N, "Total Students", BLACK, icon="👥")
    kpi(k2, f"{vc['High Risk']}", "High Risk", COL["High Risk"], f"{vc['High Risk']/N:.0%} of cohort", icon="⚠")
    kpi(k3, f"{vc['Medium Risk']}", "Medium Risk", COL["Medium Risk"], f"{vc['Medium Risk']/N:.0%} of cohort", icon="●")
    kpi(k4, f"{vc['Low Risk']}", "Low Risk", COL["Low Risk"], f"{vc['Low Risk']/N:.0%} of cohort", icon="✓")
    kpi(k5, f"{data['Attendance (%)'].mean():.0f}%", "Avg. Attendance", "#8a8", icon="◷")

    st.write("")
    c1, c2 = st.columns([1, 1.5])
    with c1:
        st.markdown('<div class="section-title">Risk Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(labels=vc.index, values=vc.values, hole=0.6, marker_colors=[COL[c] for c in vc.index],
                                textinfo="percent", textfont=dict(size=13), sort=False,
                                marker_line=dict(color="white", width=2)))
        fig.update_layout(height=270, margin=dict(l=10, r=10, t=10, b=10), showlegend=True,
                           legend=dict(orientation="h", y=-0.08, font=dict(size=11)), paper_bgcolor="rgba(0,0,0,0)", font=PLOT_FONT)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

    with c2:
        st.markdown('<div class="section-title">What drives risk across the whole cohort</div>', unsafe_allow_html=True)
        imp = pd.Series(np.abs(sv_all).mean(0), index=FEATURES).sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=BLACK,
                                text=[f"{v:.3f}" for v in imp.values], textposition="outside"))
        fig.update_layout(height=270, margin=dict(l=10, r=40, t=10, b=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           xaxis=dict(title="Mean |SHAP| (impact on risk score)", showgrid=True, gridcolor="#EEF1EA"),
                           yaxis=dict(showgrid=False), font=PLOT_FONT)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

    st.write("")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-title">Marks vs. Attendance, coloured by risk</div>', unsafe_allow_html=True)
        plot_df = data.copy(); plot_df["Average Marks"] = Xall["Average Marks"]
        fig = px.scatter(plot_df, x="Attendance (%)", y="Average Marks", color="Academic Risk",
                          color_discrete_map=COL, category_orders={"Academic Risk": CLASSES}, opacity=0.75,
                          hover_data={"Student ID": True, "Attendance (%)": True, "Average Marks": ":.0f"})
        fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="white")))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           legend=dict(orientation="h", y=-0.22, title=None, font=dict(size=11)),
                           xaxis=dict(gridcolor="#EEF1EA"), yaxis=dict(gridcolor="#EEF1EA"), font=PLOT_FONT)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

    with c4:
        st.markdown('<div class="section-title">Behaviour category vs. risk level</div>', unsafe_allow_html=True)
        ct = pd.crosstab(data["Student Behaviour"], data["Academic Risk"]).reindex(["Poor", "Average", "Good", "Excellent"]).reindex(columns=CLASSES)
        fig = go.Figure()
        for cls in CLASSES:
            fig.add_trace(go.Bar(name=cls, x=ct.index, y=ct[cls], marker_color=COL[cls]))
        fig.update_layout(barmode="stack", height=300, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           legend=dict(orientation="h", y=-0.22, font=dict(size=11)), xaxis=dict(gridcolor="#EEF1EA"),
                           yaxis=dict(title="Students", gridcolor="#EEF1EA"), font=PLOT_FONT)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

    st.write("")
    st.markdown('<div class="section-title">Students needing the most attention (ranked by P(High Risk))</div>', unsafe_allow_html=True)
    top = data_ext.sort_values("P(High Risk)", ascending=False).head(10)[
        ["Student ID", "Student Name", "Predicted Risk", "P(High Risk)", "Attendance (%)", "Student Behaviour", "Assignment Record"]]
    st.dataframe(
        top.style.format({"P(High Risk)": "{:.0%}"}).map(lambda v: f"color:{COL.get(v,'inherit')}; font-weight:700" if v in COL else "", subset=["Predicted Risk"]),
        width='stretch', hide_index=True, height=390)
    st.caption("Open **Student Explorer** to see the full explanation for any student, or **Email Center** to notify them.")

# ============================================================================ TAB 2 — STUDENT EXPLORER
with tab_explorer:
    st.sidebar.markdown("## Student Input")
    mode = st.sidebar.radio("Choose input mode", ["Pick from dataset", "Enter manually"])
    st.sidebar.markdown("---")
    if mode == "Pick from dataset":
        risk_filter = st.sidebar.multiselect("Filter by predicted risk", CLASSES, default=CLASSES, key="explorer_filter")
        options_df = directory[directory["Predicted Risk"].isin(risk_filter)]
        options = options_df["Student ID"] + " — " + options_df["Student Name"]
        if len(options) == 0:
            st.sidebar.warning("No students match this filter."); st.stop()
        sid = st.sidebar.selectbox("Select student", options, key="explorer_student")
        raw_row = directory[directory["Student ID"] == sid.split(" — ")[0]].iloc[0]
    else:
        st.sidebar.caption("Subject marks (0–100)")
        d = {}
        for c in MARK_COLS: d[c] = st.sidebar.slider(c.replace(" Marks", ""), 0, 100, 55, key=f"m_{c}")
        st.sidebar.caption("Engagement")
        d["Attendance (%)"] = st.sidebar.slider("Attendance (%)", 0, 100, 75, key="m_att")
        d["Study Hours (per day)"] = st.sidebar.slider("Study hours / day", 0, 8, 3, key="m_study")
        d["Student Behaviour"] = st.sidebar.selectbox("Behaviour", list(BEHAVIOUR), index=1, key="m_beh")
        d["Assignment Record"] = st.sidebar.selectbox("Assignment record", ["Complete", "Incomplete"], key="m_assign")
        d["Other Activities"] = st.sidebar.selectbox("Other activities", ["Present", "Not Present"], key="m_other")
        d["Student Name"] = "Manual Entry"
        raw_row = pd.Series(d)
    st.sidebar.markdown("---")
    st.sidebar.caption("Personal identifiers and gender are excluded from the model — predictions use only academic and engagement factors.")

    feats, rep = get_report(raw_row)
    pred, proba = rep["predicted_risk"], rep["probabilities"]
    name_label = raw_row.get("Student Name", "This student")
    sid_label = raw_row.get("Student ID", "")

    st.markdown(f"#### {name_label}" + (f"<span class='tag'>{sid_label}</span>" if sid_label else ""), unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.1, 1.3, 1.1])
    with c1:
        st.markdown('<div class="section-title">Predicted Risk</div>', unsafe_allow_html=True)
        st.markdown(f"<div style='text-align:center; margin: 6px 0 14px 0;'>{risk_pill(pred)}</div>", unsafe_allow_html=True)
        fig = go.Figure(go.Indicator(mode="gauge+number", value=proba["High Risk"] * 100,
            number={"suffix": "%", "font": {"size": 28, "color": INK}},
            gauge={"axis": {"range": [0, 100], "tickcolor": "#94A3B8"}, "bar": {"color": COL[pred], "thickness": 0.28}, "bgcolor": "white", "borderwidth": 0,
                   "steps": [{"range": [0, 33], "color": "#E9F6EF"}, {"range": [33, 66], "color": "#FBF1E1"}, {"range": [66, 100], "color": "#FBEAEA"}]}))
        fig.update_layout(height=190, margin=dict(l=18, r=18, t=6, b=6), paper_bgcolor="rgba(0,0,0,0)", font=PLOT_FONT)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.caption("Probability of High Risk")
    with c2:
        st.markdown('<div class="section-title">Class Probabilities</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Bar(x=list(proba.values()), y=list(proba.keys()), orientation="h", marker_color=[COL[k] for k in proba],
                                text=[f"{v:.0%}" for v in proba.values()], textposition="outside"))
        fig.update_layout(height=190, margin=dict(l=10, r=30, t=6, b=6), xaxis=dict(range=[0, 1.15], tickformat=".0%", showgrid=False),
                           yaxis=dict(showgrid=False), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, font=PLOT_FONT)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    with c3:
        st.markdown('<div class="section-title">Student Snapshot</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"<div class='metric-box'><div class='num'>{feats['Average Marks']:.0f}%</div><div class='lab'>Avg Marks</div></div>", unsafe_allow_html=True)
            st.markdown(f"<br><div class='metric-box'><div class='num'>{feats['Study Hours (per day)']:.0f} h</div><div class='lab'>Study/day</div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='metric-box'><div class='num'>{feats['Attendance (%)']:.0f}%</div><div class='lab'>Attendance</div></div>", unsafe_allow_html=True)
            st.markdown(f"<br><div class='metric-box'><div class='num'>{BEHAVIOUR_INV[int(feats['Behaviour'])]}</div><div class='lab'>Behaviour</div></div>", unsafe_allow_html=True)

    st.write("")
    c4, c5 = st.columns([1.6, 1])
    with c4:
        st.markdown('<div class="section-title">Why this prediction? (SHAP)</div>', unsafe_allow_html=True)
        s = rep["shap_risk"].sort_values()
        fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h",
                                marker_color=[COL["High Risk"] if v > 0 else COL["Low Risk"] for v in s.values],
                                text=[f"{v:+.3f}" for v in s.values], textposition="outside"))
        fig.add_vline(x=0, line_width=1, line_color="#94A3B8")
        fig.update_layout(height=260, margin=dict(l=10, r=40, t=10, b=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           xaxis=dict(title="Contribution to risk score (red = raises risk, green = lowers it)", showgrid=True, gridcolor="#EEF1EA"),
                           yaxis=dict(showgrid=False), font=PLOT_FONT)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    with c5:
        st.markdown('<div class="section-title">Risk Drivers &amp; Protective Factors</div>', unsafe_allow_html=True)
        if rep["drivers"]:
            st.markdown("**Driving risk up:**")
            st.markdown("".join(f"<span class='driver-chip'>{f}</span>" for f in rep["drivers"]), unsafe_allow_html=True)
        else:
            st.markdown("_No significant risk drivers._")
        st.write("")
        if rep["protective"]:
            st.markdown("**Protecting this student:**")
            st.markdown("".join(f"<span class='protect-chip'>{f}</span>" for f in rep["protective"]), unsafe_allow_html=True)
        else:
            st.markdown("_No strong protective factors yet._")

    st.write("")
    c6, c7 = st.columns([1.5, 1])
    with c6:
        st.markdown(f'<div class="section-title">Personalised Recommendations</div>', unsafe_allow_html=True)
        for r in rep["recommendations"]:
            st.markdown(f"<div class='rec'>{r}</div>", unsafe_allow_html=True)
    with c7:
        st.markdown('<div class="section-title">What-if: follow the plan</div>', unsafe_allow_html=True)
        if rep["after_plan"]:
            before_p, after_p = proba["High Risk"], rep["after_plan"]["p_high"]
            fig = go.Figure(go.Bar(x=["Now", "After plan"], y=[before_p, after_p], marker_color=[COL["High Risk"], COL["Low Risk"]],
                                    text=[f"{before_p:.0%}", f"{after_p:.0%}"], textposition="outside"))
            fig.update_layout(height=210, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               yaxis=dict(title="P(High Risk)", tickformat=".0%", range=[0, 1.1], showgrid=True, gridcolor="#EEF1EA"), font=PLOT_FONT)
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
            st.success(f"Predicted level moves to **{rep['after_plan']['risk']}** if these targets are met.")
            st.caption("Model-based estimate of association, not a guarantee.")
        else:
            st.info("Already on track — no plan needed.")

# ============================================================================ TAB 3 — EMAIL CENTER
with tab_email:
    st.markdown('<div class="section-title">1. Choose who to email</div>', unsafe_allow_html=True)
    ec1, ec2 = st.columns([1, 2])
    with ec1:
        risk_pick = st.multiselect("Risk level", CLASSES, default=["High Risk"], key="email_risk_pick")
    with ec2:
        pool = directory[directory["Predicted Risk"].isin(risk_pick)] if risk_pick else directory.iloc[0:0]
        names = st.multiselect("Students (leave empty to select all shown)",
                                (pool["Student ID"] + " — " + pool["Student Name"]).tolist(), key="email_names")
    recipients = pool if not names else pool[(pool["Student ID"] + " — " + pool["Student Name"]).isin(names)]
    st.caption(f"**{len(recipients)}** student(s) selected.")
    if len(recipients):
        st.dataframe(recipients[["Student ID", "Student Name", "Student Email", "Predicted Risk", "P(High Risk)"]]
                     .style.format({"P(High Risk)": "{:.0%}"}), width='stretch', hide_index=True, height=min(240, 46 + 35*len(recipients)))

    st.write("")
    st.markdown('<div class="section-title">2. Preview a draft</div>', unsafe_allow_html=True)
    if len(recipients) == 0:
        st.info("Select at least one student above to preview and send emails.")
    else:
        preview_options = (recipients["Student ID"] + " — " + recipients["Student Name"]).tolist()
        preview_pick = st.selectbox("Preview email for", preview_options, key="email_preview_pick")
        prow = recipients[(recipients["Student ID"] + " — " + recipients["Student Name"]) == preview_pick].iloc[0]
        _, prep = get_report(prow)
        draft = build_email(prow, build_features(prow.to_frame().T.astype({c: float for c in MARK_COLS + ["Attendance (%)", "Study Hours (per day)"]})).iloc[0], prep)
        pc1, pc2 = st.columns([1, 3])
        with pc1:
            st.markdown(f"{risk_pill(prep['predicted_risk'], '0.85rem')}", unsafe_allow_html=True)
            st.caption(f"To: {prow['Student Email']}")
        with pc2:
            st.markdown(f"**Subject:** {draft['subject']}")
        st.markdown(f"<div class='email-box'>{draft['body']}</div>", unsafe_allow_html=True)
        st.caption("Each recipient gets their own personalised version of the matching template (Low / Medium / High Risk) — this is just a preview of one.")

    st.write("")
    st.markdown('<div class="section-title">3. Sender settings (SMTP)</div>', unsafe_allow_html=True)
    st.caption("Credentials are kept only in this browser session and are never saved to disk. For Gmail, use a 16-character "
               "[App Password](https://myaccount.google.com/apppasswords), not your normal password.")
    sc1, sc2, sc3, sc4 = st.columns([1.3, 0.7, 1.3, 1.3])
    smtp_server = sc1.text_input("SMTP server", value=st.session_state.get("smtp_server", "smtp.gmail.com"))
    smtp_port = sc2.number_input("Port", value=st.session_state.get("smtp_port", 587), step=1)
    sender_email = sc3.text_input("Your email address", value=st.session_state.get("sender_email", ""))
    sender_pass = sc4.text_input("App password", type="password")
    mc1, mc2 = st.columns(2)
    mentor_name = mc1.text_input("Sign emails as", value="Dr. Ranjana Kale")
    institute = mc2.text_input("Institute name", value="MIT School of Computing")

    st.write("")
    st.markdown('<div class="section-title">4. Send</div>', unsafe_allow_html=True)
    test_mode = st.checkbox("Dry run (build and show the emails, but don't actually send anything)", value=True)
    confirm = st.checkbox(f"I confirm I want to {'simulate sending' if test_mode else 'SEND real emails'} to {len(recipients)} student(s).")
    send_clicked = st.button("Send Emails", type="primary", disabled=(len(recipients) == 0 or not confirm))

    if send_clicked:
        results = []
        server = None
        if not test_mode:
            if not sender_email or not sender_pass:
                st.error("Enter your email address and app password above before sending real emails.")
                st.stop()
            try:
                context = ssl.create_default_context()
                server = smtplib.SMTP(smtp_server, int(smtp_port), timeout=20)
                server.starttls(context=context)
                server.login(sender_email, sender_pass)
            except Exception as e:
                st.error(f"Could not connect/log in to {smtp_server}:{smtp_port} — {e}")
                st.stop()

        progress = st.progress(0.0, text="Preparing emails...")
        for i, (_, row) in enumerate(recipients.iterrows()):
            feats_i = build_features(row.to_frame().T.astype({c: float for c in MARK_COLS + ["Attendance (%)", "Study Hours (per day)"]})).iloc[0]
            rep_i = explain_student(model, explainer, row, feats_i)
            draft_i = build_email(row, feats_i, rep_i, mentor_name=mentor_name, institute=institute)
            status = "Prepared (dry run)"
            if not test_mode:
                try:
                    msg = MIMEText(draft_i["body"], "plain")
                    msg["Subject"], msg["From"], msg["To"] = draft_i["subject"], sender_email, row["Student Email"]
                    server.sendmail(sender_email, [row["Student Email"]], msg.as_string())
                    status = "Sent"
                    time.sleep(0.3)
                except Exception as e:
                    status = f"Failed: {e}"
            results.append({"Student ID": row["Student ID"], "Student Name": row["Student Name"], "Email": row["Student Email"],
                             "Risk": rep_i["predicted_risk"], "Subject": draft_i["subject"], "Status": status})
            progress.progress((i + 1) / len(recipients), text=f"{i+1} / {len(recipients)} processed")
        if server is not None:
            server.quit()
        progress.empty()

        res_df = pd.DataFrame(results)
        ok = res_df["Status"].isin(["Sent", "Prepared (dry run)"]).sum()
        st.success(f"{ok} / {len(res_df)} email(s) {'prepared' if test_mode else 'sent'} successfully.")
        st.dataframe(res_df, width='stretch', hide_index=True)
        st.download_button("Download this run as CSV", res_df.to_csv(index=False), "email_run.csv", "text/csv")

# ============================================================================ TAB 4 — MODEL PERFORMANCE
with tab_model:
    comp = pd.DataFrame(metrics["comparison"])
    best = comp[comp.Model == model_name].iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, model_name, "Selected Model", BLACK, icon="◆")
    kpi(k2, f"{best['CV Accuracy']:.1%}", "5-fold CV Accuracy", COL["Low Risk"], f"± {best['CV Accuracy SD']*100:.1f} pts", icon="✓")
    kpi(k3, f"{best['CV Macro-F1']:.3f}", "CV Macro-F1", "#8a8", icon="◈")
    kpi(k4, f"{metrics['external_validation']['accuracy']:.1%}", "External Validation Accuracy", COL["Medium Risk"], "on 250-student template set", icon="◷")

    st.write("")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown('<div class="section-title">Model comparison (5-fold cross-validation)</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="CV Accuracy", x=comp.Model, y=comp["CV Accuracy"], marker_color=BLACK,
                              text=[f"{v:.0%}" for v in comp["CV Accuracy"]], textposition="outside"))
        fig.add_trace(go.Bar(name="CV Macro-F1", x=comp.Model, y=comp["CV Macro-F1"], marker_color=LIME,
                              text=[f"{v:.3f}" for v in comp["CV Macro-F1"]], textposition="outside"))
        fig.update_layout(barmode="group", height=320, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           yaxis=dict(range=[0, 1.15], gridcolor="#EEF1EA"), legend=dict(orientation="h", y=-0.18), font=PLOT_FONT)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.caption(f"**{model_name}** was selected: best cross-validated macro-F1 among the tree ensembles, which also support exact SHAP attribution.")
    with c2:
        st.markdown('<div class="section-title">Hold-out confusion matrix</div>', unsafe_allow_html=True)
        cm = np.array(metrics["confusion_matrix"])
        labels = [c.replace(" Risk", "") for c in CLASSES]
        fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, colorscale=[[0, "#F4F6FA"], [1, BLACK]],
                                    text=cm, texttemplate="%{text}", textfont=dict(size=16, color=INK), showscale=False))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                           xaxis=dict(title="Predicted", side="bottom"), yaxis=dict(title="Actual", autorange="reversed"), font=PLOT_FONT)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.caption("40-student stratified hold-out set (not used in cross-validation).")

    st.write("")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-title">Global feature importance (mean |SHAP|)</div>', unsafe_allow_html=True)
        shap_imp = pd.Series(metrics["shap_mean_abs"]).sort_values()
        fig = go.Figure(go.Bar(x=shap_imp.values, y=shap_imp.index, orientation="h", marker_color=BLACK,
                                text=[f"{v:.3f}" for v in shap_imp.values], textposition="outside"))
        fig.update_layout(height=300, margin=dict(l=10, r=40, t=10, b=30), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           xaxis=dict(gridcolor="#EEF1EA"), yaxis=dict(showgrid=False), font=PLOT_FONT)
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
    with c4:
        st.markdown('<div class="section-title">Explanation fidelity check</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <p style='font-size:0.9rem; color:{INK}; line-height:1.6;'>
        The dataset's labels come from a known scoring rule, so that rule's factor weights act as ground truth.
        The SHAP importance ranking correlates strongly with those weights:</p>
        <div style='text-align:center; margin: 18px 0;'>
          <div style='font-size:2.1rem; font-weight:800; color:{INK};'>ρ = {metrics['spearman_rho']:.2f}</div>
          <div style='color:{MUTE}; font-size:0.82rem;'>Spearman correlation (p = {metrics['spearman_p']:.3f})</div>
        </div>
        <p style='font-size:0.85rem; color:{MUTE};'>This suggests the model's explanations are consistent with the true drivers of risk, not just accurate predictions.</p>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="section-title">Per-class metrics (hold-out set)</div>', unsafe_allow_html=True)
    hr = pd.DataFrame(metrics["holdout_report"]).T.loc[CLASSES, ["precision", "recall", "f1-score", "support"]]
    hr.index.name = "Risk Level"
    st.dataframe(hr.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"})
                 .background_gradient(subset=["precision", "recall", "f1-score"], cmap="Greens", vmin=0.7, vmax=1.0),
                 width='stretch')

# ============================================================================ TAB 5 — ABOUT
with tab_about:
    c1, c2 = st.columns([1.4, 1])
    with c1:
        st.markdown('<div class="section-title">About XAI-SAR</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <p style='color:{INK}; line-height:1.7; font-size:0.94rem;'>
        <b>XAI-SAR</b> (<i>An Explainable AI System for Academic Risk Prediction and Personalised Student
        Recommendations</i>) addresses a common problem: institutions rely on marks, attendance, assignment and
        engagement data, but identifying students who are academically at risk early is difficult, and traditional
        methods give little insight into <i>why</i> a student is at risk. XAI-SAR predicts a student's academic risk
        level, explains the prediction with <b>SHAP</b> (Shapley Additive Explanations), and converts that explanation
        into personalised, actionable recommendations — and, into ready-to-send emails.</p>
        <p style='color:{INK}; line-height:1.7; font-size:0.94rem;'>
        Six factors are used: average marks, attendance, study hours, behaviour, assignment completion and
        participation in other activities. Personal identifiers and gender are deliberately excluded from the model.</p>
        """, unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="section-title">How to read this dashboard</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <ul style='color:{INK}; line-height:1.9; font-size:0.92rem;'>
          <li><b>Overview</b> — cohort-wide KPIs, risk distribution, global risk drivers, and the students needing the most attention.</li>
          <li><b>Student Explorer</b> — pick any student (or enter data manually) to see their predicted risk, a SHAP-based explanation, personalised recommendations, and a what-if simulation.</li>
          <li><b>Email Center</b> — select students by risk level, preview the auto-drafted Low/Medium/High Risk email, and send it (or simulate sending) via your own email account.</li>
          <li><b>Model Performance</b> — how the model was selected, validated and how faithful its explanations are.</li>
        </ul>""", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="section-title">Project details</div>', unsafe_allow_html=True)
        rows = [("Institution", "MIT School of Computing"), ("Department", "Computer Science & Engineering"),
                ("Programme", "M.Sc. (LYMSC) — Part I"), ("Student", "Jaykumar Gaikwad"), ("Faculty Guide", "Dr. Ranjana Kale"),
                ("Dataset", f"{N} students (synthetic)"), ("Model", model_name), ("Explainability", "SHAP (TreeExplainer)")]
        for k, v in rows:
            st.markdown(f"<div style='display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid {BORDER}; font-size:0.88rem;'>"
                         f"<span style='color:{MUTE};'>{k}</span><span style='color:{INK}; font-weight:600;'>{v}</span></div>", unsafe_allow_html=True)
        st.write("")
        st.markdown('<div class="section-title">Responsible use</div>', unsafe_allow_html=True)
        st.markdown(f"""<p style='color:{MUTE}; font-size:0.85rem; line-height:1.6;'>
        Predictions are trained on synthetic data and should support a mentor's judgement, not replace it.
        The what-if simulation is a model-based estimate of association, not a guarantee of outcome.
        Before emailing real students, confirm this use complies with your institution's communication and data policies,
        and always review a draft before sending. Risk labels should not be shown to students without guidance and context.</p>""", unsafe_allow_html=True)

st.markdown(f"<p style='text-align:center; color:#9AA090; font-size:0.8rem; margin-top:24px;'>"
            f"XAI-SAR Dashboard · Model: {model_name} · For academic use only — predictions support, not replace, mentor judgement.</p>", unsafe_allow_html=True)
