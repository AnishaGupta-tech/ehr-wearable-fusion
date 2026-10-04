"""Patient Twin dashboard (Streamlit).  Run:  streamlit run dashboard/app.py
All data is synthetic. Genotypes, polygenic score and devices are SIMULATED."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from plotly.subplots import make_subplots
from config import ROOT, ANCHOR, SNPS

st.set_page_config(page_title="EHR x Wearable Fusion", layout="wide")
BLUE, ORANGE, GREEN, RED, GREY = "#2f6fb0", "#e08a1e", "#3a9a6a", "#c8453a", "#8a8f98"

profiles = [p.name for p in (ROOT / "data").iterdir() if (p / "fused_timeline.parquet").exists()]
if not profiles: st.error("No data yet. Run: python run_pipeline.py --profile sample"); st.stop()
profile = st.sidebar.selectbox("Dataset", profiles, index=0)
page = st.sidebar.radio("Page", ["Cohort", "Patient Twin", "Results"])
st.sidebar.caption("Synthetic data (Synthea + simulation). Genomics, devices and glucose traces are SIMULATED.")
D = ROOT / "data" / profile

@st.cache_data
def load_ehr(profile):
    return {k: pd.read_csv(D / "ehr" / f"{k}.csv") for k in ("patients", "labs", "conditions", "medications", "genomics")}

@st.cache_data
def load_timeline(profile, pid):
    t = pd.read_parquet(D / "timeseries.parquet", filters=[("patient_id", "==", pid)])
    return t

@st.cache_data
def load_raw(profile, pid): return pd.read_parquet(D / "raw_wearables" / f"{pid}.parquet")

ehr = load_ehr(profile); pat = ehr["patients"]

if page == "Cohort":
    st.title("Cohort overview")
    c = st.columns(4); c[0].metric("Patients", len(pat)); c[1].metric("Type 2 diabetes", int((pat.diabetes_status == "t2d").sum()))
    c[2].metric("Prediabetes", int((pat.diabetes_status == "prediabetes").sum())); c[3].metric("Median HbA1c", f"{pat.hba1c.median():.1f}%")
    a, b = st.columns(2)
    a.plotly_chart(go.Figure(go.Histogram(x=pat.hba1c, marker_color=BLUE, nbinsx=30)).update_layout(title="HbA1c (%)", height=300, margin=dict(t=40, b=10)), width="stretch")
    b.plotly_chart(go.Figure(go.Histogram(x=pat.genetic_risk_score, marker_color=ORANGE, nbinsx=25)).update_layout(title="Polygenic risk score (SIMULATED, 0-1)", height=300, margin=dict(t=40, b=10)), width="stretch")
    st.dataframe(pat.drop(columns=["birth_date"]), width="stretch", height=350)

elif page == "Patient Twin":
    st.title("Patient Twin")
    label = pat.assign(l=pat.patient_id.str[:8] + " | " + pat.diabetes_status + " | HbA1c " + pat.hba1c.astype(str) + " | " + pat.age.astype(str) + pat.sex).set_index("l").patient_id
    pid = label[st.sidebar.selectbox("Patient", label.index)]; p = pat.set_index("patient_id").loc[pid]
    tl = load_timeline(profile, pid); raw = load_raw(profile, pid)
    left, right = st.columns([1, 2.2])
    with left:
        with st.container(border=True):
            st.subheader("EHR")
            m = st.columns(3); m[0].metric("Age / sex", f"{p.age} {p.sex}"); m[1].metric("HbA1c", f"{p.hba1c}%"); m[2].metric("BMI", p.bmi)
            m = st.columns(3); m[0].metric("eGFR", p.egfr); m[1].metric("HOMA-IR", p.homa_ir); m[2].metric("Genetic risk*", p.genetic_risk_score)
            tabs = st.tabs(["Conditions", "Labs", "Medications", "Genetics", "Devices"])
            with tabs[0]:
                c = ehr["conditions"][ehr["conditions"].patient_id == pid].copy(); c["onset_date"] = pd.to_datetime(c.onset_date)
                if len(c):
                    c = c.sort_values("onset_date"); fig = go.Figure()
                    for i, r in enumerate(c.itertuples()):
                        fig.add_trace(go.Scatter(x=[r.onset_date, ANCHOR], y=[r.condition] * 2, mode="lines+markers", showlegend=False,
                                                 line=dict(color=BLUE if r.status == "active" else GREY, width=6), marker=dict(size=[10, 0]),
                                                 hovertemplate=f"{r.condition}<br>onset {r.onset_date.date()} ({r.status})<extra></extra>"))
                    fig.update_layout(height=60 + 38 * len(c), margin=dict(l=0, r=0, t=10, b=10), yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, width="stretch")
                    st.dataframe(c.assign(onset_date=c.onset_date.dt.date)[["condition", "onset_date", "status"]], hide_index=True, width="stretch")
                else: st.caption("No recorded conditions.")
            with tabs[1]:
                l = ehr["labs"][ehr["labs"].patient_id == pid].copy(); l["date"] = pd.to_datetime(l.date)
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("HbA1c (%)", "LDL (mg/dL)", "eGFR (mL/min/1.73m²)"))
                for i, (t, col) in enumerate([("HbA1c", BLUE), ("LDL", ORANGE), ("eGFR", GREEN)], 1):
                    s = l[l.test == t].sort_values("date"); fig.add_trace(go.Scatter(x=s.date, y=s.value, mode="lines+markers", line=dict(color=col), showlegend=False), row=i, col=1)
                fig.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=10)); st.plotly_chart(fig, width="stretch")
                with st.expander("All lab values"): st.dataframe(l.sort_values(["test", "date"]), hide_index=True, width="stretch")
            with tabs[2]:
                md = ehr["medications"][ehr["medications"].patient_id == pid]
                st.dataframe(md[md.status == "active"].drop(columns="patient_id"), hide_index=True, width="stretch") if (md.status == "active").any() else st.caption("No active medications.")
                if (md.status != "active").any(): st.caption("Stopped"); st.dataframe(md[md.status != "active"].drop(columns="patient_id"), hide_index=True, width="stretch")
            with tabs[3]:
                g = ehr["genomics"][ehr["genomics"].patient_id == pid].drop(columns="patient_id").rename(columns={"label": "provenance"})
                st.warning("All genotypes and the polygenic score are SIMULATED, not real genetic data.")
                st.dataframe(g, hide_index=True, width="stretch"); st.caption(f"*Weighted polygenic score (0-1): {p.genetic_risk_score}   |   top-quartile flag: {bool(p.genetic_risk_flag)}")
            with tabs[4]:
                rows = []
                for dev, d in raw.groupby("source_device"):
                    rows.append({"Device": dev, "Status": "connected", "Last sync": d.end_time.max().strftime("%Y-%m-%d %H:%M"), "Metrics": ", ".join(sorted(d.metric.unique())), "Records": len(d)})
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                q = st.columns(3); q[0].metric("CGM max gap (min)", int(tl.device_gap_minutes.max())); q[1].metric("Glucose imputed", f"{tl.glucose_imputed.mean()*100:.2f}%")
                q[2].metric("HR missing bins", f"{tl.hr_missing.mean()*100:.0f}%")
                gap = tl.device_gap_minutes > 0; st.caption(f"CGM gaps: {int((tl.device_gap_minutes.diff() < 0).sum())} dropout/warm-up episodes; glucose coverage {(tl.glucose.notna().mean()*100):.1f}% of timeline.")
    with right:
        days = sorted(tl.timestamp.dt.date.unique()); d0 = st.select_slider("Window start", days, value=days[0]); nd = st.radio("Days", [1, 2, 3, 7], index=1, horizontal=True)
        w = tl[(tl.timestamp >= pd.Timestamp(d0)) & (tl.timestamp < pd.Timestamp(d0) + pd.Timedelta(days=nd))]
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.5, 0.18, 0.18, 0.14], vertical_spacing=0.03,
                            subplot_titles=("CGM glucose (mg/dL)", "Heart rate", "Steps / 5 min", "Sleep"))
        fig.add_hrect(y0=70, y1=180, fillcolor=GREEN, opacity=0.08, line_width=0, row=1, col=1)
        fig.add_trace(go.Scatter(x=w.timestamp, y=w.glucose, line=dict(color=BLUE, width=1.6), name="glucose"), row=1, col=1)
        meals = w[w.carbs_logged > 0]; fig.add_trace(go.Scatter(x=meals.timestamp, y=[w.glucose.max() * 1.02] * len(meals), mode="markers", marker=dict(symbol="triangle-down", color=ORANGE, size=9), name="logged meal", text=meals.carbs_logged.round().astype(str) + " g"), row=1, col=1)
        fig.add_trace(go.Scatter(x=w.timestamp, y=w.hr, line=dict(color=RED, width=1), name="HR"), row=2, col=1)
        fig.add_trace(go.Bar(x=w.timestamp, y=w.steps, marker_color=GREEN, name="steps"), row=3, col=1)
        stage_color = {2: "#9ec1e6", 3: "#2f6fb0", 4: "#8e6bbf", 1: "#e0b050"}
        for code, col in stage_color.items():
            s = w[w.sleep_stage == code]; fig.add_trace(go.Bar(x=s.timestamp, y=[1] * len(s), marker_color=col, name={1: "Awake", 2: "Core", 3: "Deep", 4: "REM"}[code], width=300000), row=4, col=1)
        fig.update_layout(height=640, margin=dict(l=0, r=0, t=30, b=10), bargap=0, barmode="stack", legend=dict(orientation="h", y=-0.05))
        fig.update_yaxes(visible=False, row=4, col=1); st.plotly_chart(fig, width="stretch")
        n = tl[tl.timestamp >= pd.Timestamp(d0)].iloc[0]
        s = st.columns(5); s[0].metric("Resting HR", n.resting_hr if pd.notna(n.resting_hr) else "–"); s[1].metric("Sleep (h)", round(n.sleep_hours_last_night, 1) if pd.notna(n.sleep_hours_last_night) else "–")
        s[2].metric("SpO2 night min", n.spo2_night_min if pd.notna(n.spo2_night_min) else "–"); s[3].metric("Wrist temp Δ", n.wrist_temp_dev if pd.notna(n.wrist_temp_dev) else "–"); s[4].metric("HRV (ms)", round(n.hrv_rmssd_latest, 0))

else:
    st.title("Ablation results: does EHR help predict glucose 60 min ahead?")
    rf = D / "results" / "ablation.csv"
    if not rf.exists(): st.info("Run the models stage first."); st.stop()
    res = pd.read_csv(rf, index_col=0); meta = json.load(open(D / "results" / "ablation.json"))
    st.caption(f"{meta['n_patients']} patients, {meta['n_rows']:,} evaluation rows, patient-grouped 5-fold CV, LightGBM. Lower MAE/RMSE is better.")
    fig = go.Figure(go.Bar(x=res.index, y=res.MAE, marker_color=[GREY, BLUE, ORANGE, GREEN][:len(res)], text=res.MAE.round(2), textposition="outside"))
    fig.update_layout(title="MAE (mg/dL)", height=360, margin=dict(t=40)); st.plotly_chart(fig, width="stretch")
    st.dataframe(res.round(3), width="stretch")
    st.subheader("Paired per-patient tests (MAE gain, mg/dL)"); st.dataframe(pd.DataFrame(meta["paired_tests"]).T.round(4), width="stretch")
    st.subheader("Top features (gain)")
    for name, imp in meta["top_features"].items():
        if name == "Wearable-only": continue
        s = pd.Series(imp).sort_values(); st.plotly_chart(go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=BLUE)).update_layout(title=name, height=380, margin=dict(t=40, l=0)), width="stretch")
