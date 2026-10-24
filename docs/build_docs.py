"""Builds docs/architecture.pdf and docs/TeamTwins.pdf (matplotlib only)."""
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).parent
NAVY, CYAN, GREEN, ORANGE = "#15347c", "#1fb5c4", "#25c06d", "#e8891c"
LB, LG, LC, INK, GREY = "#dcecfb", "#e2f7ea", "#d9f3f6", "#10213f", "#55607a"
W, H = 13.33, 7.5
SECTIONS = ["Problem", "Organiser", "Data", "Pipeline", "Models", "Results", "Dashboard", "Impact", "Roadmap"]
PAGE = [0]


def wrap(t, n):
    return textwrap.wrap(str(t), max(8, int(n)))


def rbox(ax, x, y, w, h, fc, ec="none", r=0.1, lw=1.0, ls="-", z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def txt(ax, x, y, s, fs=10, color=INK, weight="normal", ha="left", va="center", z=5, **kw):
    ax.text(x, y, s, fontsize=fs, color=color, weight=weight, ha=ha, va=va, zorder=z, **kw)


def para(ax, x, y, s, width_in, fs=10, color=INK, weight="normal", gap=1.45):
    for ln in wrap(s, width_in * 122 / fs):
        txt(ax, x, y, ln, fs, color, weight); y -= fs / 72 * gap
    return y


def fitpara(ax, x, ytop, ybottom, width_in, s, color=INK, weight="normal", maxfs=15, minfs=8):
    fs = maxfs
    while fs > minfs and len(wrap(s, width_in * 122 / fs)) * fs / 72 * 1.5 > ytop - ybottom:
        fs -= 0.5
    ls = wrap(s, width_in * 122 / fs); gap = min(fs / 72 * 1.6, (ytop - ybottom) / max(len(ls), 1))
    y = ytop
    for ln in ls:
        txt(ax, x, y, ln, fs, color, weight); y -= gap
    return y


def nav(ax, active):
    n = len(SECTIONS); w = (W - 0.3) / n
    for i, name in enumerate(SECTIONS):
        x = 0.15 + i * w; y, h = 6.98, 0.42
        pts = [(x, y), (x + w - 0.14, y), (x + w, y + h / 2), (x + w - 0.14, y + h), (x, y + h), (x + 0.14, y + h / 2)]
        ax.add_patch(Polygon(pts, closed=True, fc=GREEN if i == active else NAVY, ec="white", lw=1.5, zorder=3))
        txt(ax, x + w / 2 + 0.05, y + h / 2, name, 8.5, "white", "bold", "center")


def base(active, num=True):
    PAGE[0] += 1
    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    nav(ax, active)
    txt(ax, 0.2, 0.14, "TeamTwins  |  Glucose Digital Twin  |  Digital Twin Challenge 2026, Happiest Health", 7.5, GREY)
    txt(ax, W - 0.2, 0.14, f"{PAGE[0]:02d}", 8, GREY, "bold", "right")
    return fig, ax


def panel(ax, x, y, w, h, title, fill=LB, tab=NAVY):
    rbox(ax, x, y, w, h, fill, NAVY, 0.12, 1.2, (0, (4, 3)), 1)
    tw = len(title) * 0.115 + 0.45
    rbox(ax, x + 0.15, y + h - 0.2, tw, 0.4, tab, "none", 0.06, z=4)
    txt(ax, x + 0.15 + tw / 2, y + h, title, 11, "white", "bold", "center")


def circle_stat(ax, x, y, num, label, w=2.6, fs=15):
    rbox(ax, x + 0.4, y - 0.27, w, 0.54, "#d2d8f0", "none", 0.1, z=2)
    ax.add_patch(Circle((x + 0.4, y), 0.42, fc=NAVY, ec=CYAN, lw=3, zorder=4))
    txt(ax, x + 0.4, y, num, fs, "white", "bold", "center")
    txt(ax, x + 1.0, y, label, min(11, (w - 0.75) * 118 / max(len(label), 1)), INK, "bold")


def hbars(ax, x, ytop, w, items, maxv, color=NAVY, rowh=0.3, fs=9, labw=1.7, fmt="{:g}"):
    for i, (lab, v) in enumerate(items):
        y = ytop - i * rowh
        txt(ax, x, y, lab, fs, INK, ha="left")
        bw = (w - labw - 0.5) * v / maxv
        rbox(ax, x + labw, y - 0.09, max(bw, 0.05), 0.18, color, "none", 0.05, z=3)
        txt(ax, x + labw + bw + 0.07, y, fmt.format(v), fs, GREY, "bold")


def pill(ax, x, y, w, label, pct, color=NAVY, fs=9.5):
    rbox(ax, x, y - 0.13, w, 0.26, "#e1e4ea", "none", 0.12, z=2)
    rbox(ax, x, y - 0.13, max(w * pct / 100, 0.3), 0.26, color, "none", 0.12, z=3)
    txt(ax, x + max(w * pct / 100, 0.3) - 0.08, y, f"{pct:g}%", fs - 0.5, "white", "bold", "right")
    txt(ax, x + w + 0.12, y, label, fs, INK, "bold")


def gauge(ax, cx, cy, r, pct, label, color=NAVY, fs=13):
    ax.add_patch(Wedge((cx, cy), r, 0, 360, width=r * 0.3, fc="#dfe3ea", ec="none", zorder=3))
    ax.add_patch(Wedge((cx, cy), r, 90 - 3.6 * pct, 90, width=r * 0.3, fc=color, ec="none", zorder=4))
    txt(ax, cx, cy, f"{pct:g}%", fs, INK, "bold", "center")
    txt(ax, cx, cy - r - 0.2, label, 8.5, INK, "bold", "center")


def donut(ax, cx, cy, r, parts, colors):
    tot = sum(v for _, v in parts); a = 90
    for (lab, v), c in zip(parts, colors):
        ang = 360 * v / tot
        ax.add_patch(Wedge((cx, cy), r, a - ang, a, width=r * 0.42, fc=c, ec="white", lw=1.5, zorder=3))
        mid = np.radians(a - ang / 2)
        txt(ax, cx + (r * 1.28) * np.cos(mid), cy + (r * 1.28) * np.sin(mid), f"{lab}\n{v}", 8, INK, "bold", "center")
        a -= ang


def chart(fig, x, y, w, h):
    a = fig.add_axes([x / W, y / H, w / W, h / H]); a.set_facecolor("none")
    a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=8.5)
    return a


def table(ax, x, ytop, widths, header, rows, fs=9, pad=0.06, hl=None, hdr=NAVY, bold_first=True, until=None):
    def _nat(f, pd_):
        l_ = f / 72 * 1.4; c_ = lambda t, wd: wrap(t, (wd - 0.18) * 118 / f) or [""]
        return l_ * max(len(c_(h, wd)) for h, wd in zip(header, widths)) + 2 * pd_ + sum(l_ * max(len(c_(c, wd)) for c, wd in zip(r, widths)) + 2 * pd_ for r in rows)
    if until is not None:
        f = 12.0
        while f > fs and _nat(f, 0.05) > ytop - until: f -= 0.25
        fs = max(f, fs); pad = 0.05
        pad += max(0.0, min(0.2, (ytop - until - _nat(fs, pad)) / (2 * (len(rows) + 1))))
    xs = [x]
    for wd in widths: xs.append(xs[-1] + wd)
    lh = fs / 72 * 1.4
    cw = lambda t, wd: wrap(t, (wd - 0.18) * 118 / fs) or [""]
    hh = lh * max(len(cw(h, wd)) for h, wd in zip(header, widths)) + 2 * pad
    ax.add_patch(Rectangle((x, ytop - hh), sum(widths), hh, fc=hdr, ec="white", lw=1, zorder=3))
    for i, h_ in enumerate(header):
        for k, ln in enumerate(cw(h_, widths[i])):
            txt(ax, xs[i] + 0.09, ytop - pad - lh * (k + 0.5), ln, fs, "white", "bold")
    y = ytop - hh
    for r, row in enumerate(rows):
        cells = [cw(c, wd) for c, wd in zip(row, widths)]
        rh = lh * max(len(c) for c in cells) + 2 * pad
        on = hl is not None and r in hl
        ax.add_patch(Rectangle((x, y - rh), sum(widths), rh, fc="#cdeedc" if on else ("white" if r % 2 else "#eef4fb"), ec="#c7d3e6", lw=0.6, zorder=2))
        for i, c in enumerate(cells):
            for k, ln in enumerate(c):
                txt(ax, xs[i] + 0.09, y - pad - lh * (k + 0.5), ln, fs, INK, "bold" if (on or (i == 0 and bold_first)) else "normal")
        y -= rh
    return y


def banner(ax, x, y, w, h, lead, items):
    rbox(ax, x, y, w, h, NAVY, "none", 0.12, z=2)
    txt(ax, x + 0.25, y + h / 2, lead, 11, "white", "bold")
    x0 = x + 2.7; step = (x + w - x0) / len(items)
    for i, (t, s) in enumerate(items):
        xx = x0 + i * step
        txt(ax, xx, y + h * 0.68, t, 10.5, "#9ff0c1", "bold")
        for k, ln in enumerate(wrap(s, (step - 0.3) * 118 / 8.5)[:2]):
            txt(ax, xx, y + h * 0.36 - k * 0.15, ln, 8.5, "white")


def tile(ax, x, y, w, h, title, body, color=NAVY, fs=9):
    rbox(ax, x, y, w, h, "white", color, 0.08, 1.4, z=2)
    rbox(ax, x, y + h - 0.34, w, 0.34, color, "none", 0.08, z=3)
    txt(ax, x + w / 2, y + h - 0.17, title, 9.5, "white", "bold", "center")
    avail = h - 0.5; f = 13.0
    while f > fs and len(wrap(body, (w - 0.22) * 122 / f)) * f / 72 * 1.45 > avail:
        f -= 0.5
    ls = wrap(body, (w - 0.22) * 122 / f); gap = f / 72 * 1.45
    yy = y + h - 0.5 - (avail - len(ls) * gap) / 4 - gap / 2
    for ln in ls:
        txt(ax, x + 0.11, yy, ln, f, INK); yy -= gap


def arrow(ax, x1, y1, x2, y2, c=GREEN, lw=2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=lw, color=c, zorder=6))


def avatar(ax, x, y, initials, r=0.5, c=NAVY):
    ax.add_patch(Circle((x, y), r, fc=c, ec="white", lw=3, zorder=4)); txt(ax, x, y, initials, 20, "white", "bold", "center")


def arch_draw(ax):
    def box(x, y, w, h, t, fc=LB, fs=9.5):
        rbox(ax, x, y, w, h, fc, NAVY, 0.1, 1.3)
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.35, zorder=5)
    ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis("off")
    box(0.2, 5.3, 2.6, 1.3, "Synthea\nsynthetic patients"); box(3.4, 5.3, 2.9, 1.3, "EHR tables\nlabs, diagnoses, medicines,\nSNPs  (ehr/build_ehr.py)")
    box(6.9, 5.3, 3.1, 1.3, "EHR to physiology\nwearables/derive_params.py"); box(10.6, 5.3, 3.2, 1.3, "Simulators\nsimglucose CGM +\nwatch streams")
    box(0.2, 3.1, 3.1, 1.3, "Raw device streams\nnative rates, gaps,\nunder logged meals"); box(3.9, 3.1, 3.1, 1.3, "Ingestion and alignment\n5 minute timeline\n(ingestion/align.py)")
    box(7.6, 3.1, 3.1, 1.3, "Fused patient timeline\nEHR joined only from\nlabs dated before t"); box(11.3, 3.1, 2.5, 1.3, "Features\n(features.py)")
    box(0.2, 0.7, 3.9, 1.4, "Event models (LightGBM)\nP(>180), P(<70) in 2 h\n1 h and 2 h forecast bands", "#fdf1e3")
    box(4.7, 0.7, 3.4, 1.4, "Alerts and SHAP\n3 sensitivities\nscore_stream, no look ahead", "#fdf1e3")
    box(8.7, 0.7, 5.1, 1.4, "Doctor dashboard (Streamlit)\nTriage board, patient view + what if\nlive monitor, model results", "#fdf1e3")
    for a in [(2.8, 5.95, 3.4, 5.95), (6.3, 5.95, 6.9, 5.95), (10.0, 5.95, 10.6, 5.95), (12.2, 5.3, 2.5, 4.4), (3.3, 3.75, 3.9, 3.75), (7.0, 3.75, 7.6, 3.75),
              (10.7, 3.75, 11.3, 3.75), (4.9, 5.3, 9.0, 4.4), (12.5, 3.1, 2.5, 2.1), (4.1, 1.4, 4.7, 1.4), (8.1, 1.4, 8.7, 1.4)]:
        arrow(ax, *a)


def arch_detail(ax):
    ax.set_xlim(0, 16); ax.set_ylim(0, 8.95); ax.axis("off")
    CREAM, GR = "#fdf1e3", "#eef0f4"

    def box(x, y, w, h, title, sub, kind="code"):
        fc, ec, tc, ls = {"code": (LB, NAVY, NAVY, "-"), "data": (CREAM, ORANGE, "#9a5a0c", (0, (3, 2))), "ops": (GR, "#8a93a6", "#4a5568", (0, (2, 2)))}[kind]
        rbox(ax, x, y, w, h, fc, ec, 0.08, 1.3, ls, 3)
        ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="center", fontsize=9, weight="bold", color=tc, zorder=6)
        ax.text(x + w / 2, y + (h - 0.38) / 2, sub, ha="center", va="center", fontsize=7.6, color=INK, linespacing=1.3, zorder=6)

    def arr(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.8, color=GREEN, zorder=7))

    def chips(y, items, x0=0.8):
        x = x0
        for t_ in items:
            wd = len(t_) * 0.082 + 0.3
            rbox(ax, x, y, wd, 0.2, "white", "#b9c6dc", 0.1, 0.8, z=4)
            ax.text(x + wd / 2, y + 0.1, t_, ha="center", va="center", fontsize=6.8, color=GREY, weight="bold", zorder=6); x += wd + 0.12

    names = [("1  DATA", NAVY), ("2  SIMULATION", CYAN), ("3  INGESTION", NAVY), ("4  MODELS", CYAN), ("5  SERVING", NAVY)]
    bot = [0.55 + (4 - k) * 1.62 for k in range(5)]
    for k, (lab, c) in enumerate(names):
        ax.add_patch(Rectangle((0.1, bot[k] - 0.02), 15.85, 1.56, fc="#f3f8fd", ec="#c9d8ee", lw=0.8, zorder=1))
        ax.add_patch(Rectangle((0.1, bot[k] - 0.02), 0.5, 1.56, fc=c, ec="none", zorder=2))
        ax.text(0.35, bot[k] + 0.76, lab, rotation=90, ha="center", va="center", fontsize=8.5, weight="bold", color="white", zorder=6)
    by = [b_ + 0.3 for b_ in bot]; H_ = 1.1
    # lane 1 (left to right)
    box(0.8, by[0], 2.3, H_, "run_pipeline.py", "orchestrates stages:\nsynthea, ehr, wearables,\ningest, models, events", "ops"); box(3.3, by[0], 2.3, H_, "config.py", "profiles: sample 50 x 7 d,\nfull 800 x 14 d\nANCHOR 2026-09-30", "ops")
    box(5.8, by[0], 2.1, H_, "Synthea", "synthetic patients\ndemographics, dx dates,\nlipids, smoking"); box(8.3, by[0], 3.4, H_, "ehr/build_ehr.py", "labs (4 to 8 visits), fasting insulin,\nmedicines, complications,\n5 SNPs and polygenic score")
    box(12.1, by[0], 3.7, H_, "data/<profile>/ehr/*.csv", "patients, conditions, labs,\nmedications, genomics\n(genomics labelled SIMULATED)", "data")
    arr(7.9, by[0] + 0.55, 8.3, by[0] + 0.55); arr(11.7, by[0] + 0.55, 12.1, by[0] + 0.55)
    chips(bot[0] + 0.03, ["Synthea Apache 2.0, Java 17", "5 CSV tables", "labs: HbA1c, glucose, insulin, lipids, eGFR, ALT"])
    # lane 2 (right to left)
    box(11.5, by[1], 4.3, H_, "wearables/derive_params.py", "EHR to physiology: Vmx, kp3, k1, m4, kp1,\nsecretion, renal threshold, drug effects;\ncalibrated to eAG from HbA1c"); box(6.4, by[1], 4.7, H_, "wearables/simulate.py", "simglucose UVA/Padova ODE + PI beta cell\nbehaviour: meals, steps, sleep, heart rate,\nHRV, SpO2, wrist temperature")
    box(3.9, by[1], 2.2, H_, "wearables/generate.py", "per patient run\nwrites sim_params.csv"); box(0.8, by[1], 2.8, H_, "raw_wearables, truth", "native rates, gaps, dropouts,\nunder logged meals;\ntruth = validation only", "data")
    arr(13.6, by[0], 13.6, by[1] + H_); arr(11.5, by[1] + 0.55, 11.1, by[1] + 0.55); arr(6.4, by[1] + 0.55, 6.1, by[1] + 0.55); arr(3.9, by[1] + 0.55, 3.6, by[1] + 0.55)
    chips(bot[1] + 0.03, ["CGM every 5 min", "meal logging p 0.35 to 0.95", "15 to 120 min dropouts", "1 minute truth kept"], 2.7)
    # lane 3 (left to right)
    box(0.8, by[2], 2.6, H_, "load_raw_streams", "one long format parquet\nper patient: metric,\nstart, end, value, unit"); box(3.8, by[2], 4.0, H_, "ingestion/align.py: align_to_timeline", "nearest reading within 2.5 min\ninterpolate gaps of 15 min or less\nimputed and gap flags; daily metrics")
    box(8.2, by[2], 2.3, H_, "timeseries.parquet", "aligned, pre fusion", "data"); box(10.9, by[2], 2.9, H_, "fuse_with_ehr", "labs, conditions, medicines\ndated before the day only\ntest: no future lab leakage"); box(14.2, by[2], 1.6, H_, "fused_timeline", "EHR + wearables\n5 min grid", "data")
    arr(2.2, by[1], 2.2, by[2] + H_); arr(3.4, by[2] + 0.55, 3.8, by[2] + 0.55); arr(7.8, by[2] + 0.55, 8.2, by[2] + 0.55); arr(10.5, by[2] + 0.55, 10.9, by[2] + 0.55); arr(13.8, by[2] + 0.55, 14.2, by[2] + 0.55)
    chips(bot[2] + 0.03, ["5 minute grid", "gaps over 15 min stay missing", "hr_missing and glucose_imputed flags", "leakage unit test"])
    # lane 4 (right to left)
    box(11.3, by[3], 4.5, H_, "features.py and models/ablation.py", "102 causal features: glucose dynamics,\ninsulin and carbs on board, personal history,\nphysiology, health record; 60 min ablation"); box(4.9, by[3], 6.0, H_, "models/events.py", "LightGBM: P(>180) and P(<70) within 2 h; quantile forecasts +60 and +120 min,\nconformal 80% band; thresholds sensitive, balanced, precise;\npatient grouped 5 fold CV; per fold boosters kept for SHAP")
    box(0.8, by[3], 3.7, H_, "results/*", "events.csv and json, event_risk.parquet,\nablation.csv and json,\nevent_models/*.txt boosters", "data")
    arr(15.0, by[2], 15.0, by[3] + H_); arr(11.3, by[3] + 0.55, 10.9, by[3] + 0.55); arr(4.9, by[3] + 0.55, 4.5, by[3] + 0.55)
    chips(bot[3] + 0.03, ["patient grouped 5 fold CV", "conformal 80% band", "paired Wilcoxon ablation", "SHAP from booster contributions"], 2.7)
    # lane 5 (left to right)
    box(0.8, by[4], 3.6, H_, "models/events.py: score_stream", "rebuild features from history so far\nscore with held out fold model\nno look ahead (tested)"); box(4.8, by[4], 2.9, H_, "alert rule", "risk above threshold now\nand 10 minutes earlier\nthree alert settings")
    box(8.1, by[4], 5.3, H_, "dashboard/app.py (Streamlit)", "Triage board ranks the panel; Patient view with SHAP and what if;\nLive monitor with alert log; Model results page"); box(13.8, by[4], 2.0, H_, "tests (13)", "leakage, labels,\nstreaming = batch", "data")
    arr(2.2, by[3], 2.2, by[4] + H_); arr(4.4, by[4] + 0.55, 4.8, by[4] + 0.55); arr(7.7, by[4] + 0.55, 8.1, by[4] + 0.55)
    chips(bot[4] + 0.03, ["Python 3.12", "pytest", "Streamlit + Plotly", "LightGBM, scikit-learn"])
    ax.text(0.1, 0.22, "Blue = code module     Cream dashed = data artifact     Grey dashed = orchestration     Green arrow = data flow", fontsize=8, color=GREY, zorder=6)


def architecture():
    fig = plt.figure(figsize=(16, 9.8)); fig.patch.set_facecolor("white")
    fig.text(0.5, 0.96, "Architecture: EHR x wearable fusion digital twin (all data synthetic)", ha="center", fontsize=17, weight="bold", color=NAVY)
    fig.text(0.5, 0.93, "Five layers, read top to bottom and snaking left to right: each box names the file that implements it", ha="center", fontsize=10.5, color=GREY)
    arch_detail(fig.add_axes([0.01, 0.01, 0.98, 0.91]))
    return fig


def real_data():
    r = pd.read_parquet(ROOT / "data/sample/results/event_risk.parquet")
    last = r.dropna(subset=["p_hyper"]).sort_values("timestamp").groupby("patient_id").tail(1)
    top = last.sort_values("p_hyper", ascending=False).head(8).reset_index(drop=True)
    pid = r.loc[r.p_hyper.idxmax(), "patient_id"]
    d = r[r.patient_id == pid].dropna(subset=["glucose"]).copy()
    t0 = d.loc[d.p_hyper.idxmax(), "timestamp"]
    d = d[(d.timestamp >= t0 - pd.Timedelta(hours=18)) & (d.timestamp <= t0 + pd.Timedelta(hours=6))]
    return top, d, str(pid)[:8]


def title_slide():
    PAGE[0] += 1
    fig = plt.figure(figsize=(W, H)); fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.add_patch(Rectangle((0, 1.9), W, H - 1.9, fc=NAVY, ec="none"))
    ax.add_patch(Polygon([(W, 1.9), (W, 4.6), (W - 2.6, 1.9)], fc="white", ec="none"))
    txt(ax, 0.5, 6.95, "DIGITAL TWIN CHALLENGE 2026", 13, "white", "bold")
    txt(ax, 0.5, 6.6, "Reimagining and Reforming Healthcare in India Summit 2026", 10.5, "#b8c7ee")
    txt(ax, W - 0.5, 6.95, "HAPPIEST HEALTH", 17, "white", "bold", "right")
    txt(ax, W - 0.5, 6.6, "#ReimaginingHealthcare", 10, "#b8c7ee", ha="right")
    txt(ax, W / 2, 5.0, "GLUCOSE", 58, "white", "bold", "center"); txt(ax, W / 2, 3.95, "DIGITAL TWIN", 58, "#7ff0b5", "bold", "center")
    txt(ax, W / 2, 2.95, "Fusing health records and wearable streams to warn of high and low glucose two hours ahead", 14, "white", ha="center")
    txt(ax, W / 2, 2.5, "A type 2 diabetes proof of concept. All data synthetic.", 11, "#b8c7ee", ha="center")
    avatar(ax, 1.2, 0.95, "AG"); txt(ax, 1.95, 1.15, "Anisha Gupta (Team Leader)", 12, INK, "bold"); txt(ax, 1.95, 0.85, "Indira Gandhi Delhi Technical\nUniversity for Women", 9.5, GREY, va="top")
    avatar(ax, 5.9, 0.95, "VS", c=CYAN); txt(ax, 6.65, 1.15, "Vibhor Sharma", 12, INK, "bold"); txt(ax, 6.65, 0.85, "GL Bajaj Institute of\nTechnology and Management", 9.5, GREY, va="top")
    txt(ax, 11.6, 1.3, "Team", 20, INK, "bold", "center"); txt(ax, 11.3, 0.8, "“TeamTwins”", 22, NAVY, "bold", "center")
    return fig


def build():
    F = [title_slide()]
    top, d, pid = real_data()

    # 2 problem
    fig, ax = base(0)
    panel(ax, 0.2, 0.5, 4.3, 6.2, "Why this matters")
    for i, (n, l) in enumerate([("101 M", "people with diabetes in India"), ("136 M", "people with prediabetes"), ("5 min", "CGM reading interval"), ("2 h", "warning time targeted"), ("800", "synthetic patients, 14 days")]):
        circle_stat(ax, 0.55, 5.9 - i * 1.0, n, l, w=3.3)
    para(ax, 0.5, 0.95, "ICMR INDIAB 2023 for prevalence. South Asians develop type 2 diabetes at lower BMI and younger ages.", 3.8, 8.5, GREY)
    panel(ax, 4.7, 2.6, 8.4, 4.1, "Where care breaks down", LG)
    cards = [("Seen rarely", "A patient meets a clinician a few times a year, so a high above 180 or a low below 70 between visits is never seen."),
             ("Data unlinked", "CGMs and smartwatches stream every few minutes, but the streams are not joined to labs, diagnoses, medicines or genetics."),
             ("Too much to read", "No doctor can scan every trace, so risk is found after the event rather than before it."),
             ("No priority order", "Without ranking, a clinic cannot tell which patient needs attention first.")]
    for i, (t, b) in enumerate(cards):
        tile(ax, 4.95 + (i % 2) * 4.1, 4.15 - (i // 2) * 1.7, 3.9, 1.55, t, b, [NAVY, CYAN, CYAN, NAVY][i], 9.5)
    for i, t_ in enumerate(["Event: 15 min out of range", "High above 180, low below 70 mg/dL", "Horizon: up to 2 hours ahead"]):
        rbox(ax, 4.95 + i * 2.78, 5.85, 2.65, 0.5, NAVY, "none", 0.1, z=3); txt(ax, 4.95 + i * 2.78 + 1.325, 6.1, t_, 8.8, "white", "bold", "center", z=6)
    panel(ax, 4.7, 0.5, 8.4, 1.9, "The question we answer", LC)
    fitpara(ax, 5.0, 2.1, 0.7, 7.9, "Can a patient's record (labs, diagnoses, medicines, genetics) fused with wearable streams warn of a glucose event above 180 or below 70 mg/dL up to two hours ahead, explain why, and rank the whole panel by who needs attention first?", INK, "bold", 15)
    F.append(fig)

    # A clinical definitions
    fig, ax = base(0)
    panel(ax, 0.2, 0.5, 4.6, 6.25, "Glucose zones (mg/dL)")
    zones = [(300, 180, ORANGE, "Level: high  >180"), (180, 70, GREEN, "In range  70 to 180"), (70, 54, "#e74c3c", "Level 1 low  <70"), (54, 20, "#8e1b1b", "Level 2 low  <54")]
    ytop = 6.1; scale = 0.0165
    for hi, lo, c, lab in zones:
        hgt = (min(hi, 300) - lo) * scale
        ax.add_patch(Rectangle((0.6, ytop - hgt), 1.3, hgt, fc=c, ec="white", lw=2, zorder=3)); txt(ax, 2.05, ytop - hgt / 2, lab, 10, INK, "bold"); ytop -= hgt
    para(ax, 0.45, 0.95, "Thresholds follow the international consensus on CGM metrics: time in range 70 to 180 mg/dL, with level 1 low below 70 and level 2 low below 54.", 4.1, 9.3, GREY)
    panel(ax, 5.0, 3.4, 8.1, 3.35, "How we define an event", LG)
    for i, (t_, b_, c) in enumerate([("Out of range for 15 min", "an event is at least 3 consecutive 5 minute readings above 180 or below 70, so single sample sensor spikes are ignored", NAVY), ("Onset after an hour in range", "an onset counts only after one hour fully in range, so we predict new events, not ones already underway", CYAN), ("Look ahead 5 to 120 min", "the label looks only at t+5 to t+120 minutes; features at t use data up to t", NAVY), ("Alert needs confirmation", "risk must exceed the threshold now and 10 minutes earlier (2 readings)", GREEN)]):
        tile(ax, 5.2 + (i % 2) * 3.9, 5.0 - (i // 2) * 1.55, 3.75, 1.4, t_, b_, c, 9)
    panel(ax, 5.0, 0.5, 8.1, 2.65, "Why these choices matter clinically")
    for i, (n, l) in enumerate([("15 min", "sustained excursion, not noise"), ("2 h", "time to act: snack, dose, call"), ("3 levels", "sensitive, balanced, precise"), ("1 h", "in range before an onset counts")]):
        circle_stat(ax, 5.2 + (i % 2) * 3.9, 2.4 - (i // 2) * 0.85, n, l, w=3.15, fs=10)
    para(ax, 5.2, 0.8, "Hypoglycaemia is rare (2.7% of in range moments), so every precision figure is read against that base rate.", 7.6, 9.5, GREY, "bold")
    F.append(fig)

    # 3 organiser
    fig, ax = base(1)
    panel(ax, 0.2, 3.3, 6.5, 3.45, "The organiser: Happiest Health")
    para(ax, 0.45, 6.2, "An integrated wellness enterprise based in Koramangala, Bengaluru (Happiest Health Systems Pvt. Ltd.), announced in April 2022 and promoted by Ashok Soota. Mission: inspire everyone to embrace wellness as a way of life.", 6.0, 10.5)
    for i, (t, b, c) in enumerate([("Knowledge", "Magazine, daily Healthzine, podcasts, videos, summits", NAVY), ("Diagnostics", "Routine and specialised blood tests, Bengaluru", CYAN), ("Healthcare", "Dental and orthopaedics now; ENT, eye, paediatrics, obesity, urology planned", NAVY), ("Wellness", "Ayurveda, mental health, physio; Wellness Catalyst for corporates", CYAN)]):
        tile(ax, 0.4 + i * 1.57, 3.5, 1.5, 1.75, t, b, c, 8.2)
    panel(ax, 6.9, 3.3, 6.2, 3.45, "Ashok Soota, promoter", LG)
    for i, (n, l) in enumerate([("Wipro", "grew its IT business into India's largest computer company"), ("2007", "founding Chairman, MindTree (IPO)"), ("2020", "founding Chairman, Happiest Minds (IPO)"), ("2021", "founded SKAN: ageing and neurological research")]):
        circle_stat(ax, 7.1, 6.0 - i * 0.62, n, l, w=5.2, fs=9.5)
    para(ax, 7.1, 3.62, "Ashirvadam trust (2011). Forbes Asia Top 15 philanthropists, 2022.", 5.8, 8.8, GREY)
    panel(ax, 0.2, 0.5, 8.0, 2.55, "The challenge: Digital Twin Challenge 2026", LC)
    for i, (d_, t, c) in enumerate([("18 Sep", "Submissions open", NAVY), ("24 Oct", "Code due, 7 PM IST", GREEN), ("Date TBA", "Top 10 jury presentation (virtual)", CYAN), ("20 Nov", "Top 5 showcase, Bengaluru", NAVY)]):
        x = 0.5 + i * 1.95
        ax.add_patch(Circle((x + 0.15, 1.95), 0.15, fc=c, ec="white", lw=2, zorder=5)); txt(ax, x + 0.4, 1.95, d_, 10.5, c if c != CYAN else NAVY, "bold")
        para(ax, x + 0.4, 1.68, t, 1.5, 10)
    ax.plot([0.65, 7.7], [1.95, 1.95], color=NAVY, lw=1.5, zorder=4)
    para(ax, 0.45, 1.05, "Prizes: Rs 1,00,000 winner, Rs 50,000 first runner up, Rs 30,000 second runner up, two Rs 10,000 recognition awards, plus internship (PPI) opportunities. Travel and stay for two members of each finalist team are sponsored.", 7.5, 9.6, GREY)
    panel(ax, 8.4, 0.5, 4.7, 2.55, "Why our project fits")
    fitpara(ax, 8.65, 2.5, 1.35, 4.2, "The summit programme centres on evidence based care, continuous health modelling and the opportunities and limits of AI. Our twin is exactly a continuous health model, with honest limits stated.", INK, "normal", 13)
    para(ax, 8.65, 1.1, "Sources: happiesthealth.com/about-us, district.in event listing, Unstop challenge page.", 4.2, 7.5, GREY)
    F.append(fig)

    # 4 requirements
    fig, ax = base(1)
    panel(ax, 0.2, 0.5, 8.6, 6.2, "Challenge brief and where each point is met")
    table(ax, 0.4, 6.35, [3.0, 3.9, 1.5], ["Requirement", "How this submission meets it", "Where"], [
        ["Static data: simulated EHR with demographics, diagnoses, labs, genetic markers", "Synthea plus generated labs, medicines, 5 SNPs and a polygenic score", "ehr/build_ehr.py"],
        ["Dynamic data: wearable or CGM time series (HRV, glucose, sleep, steps)", "Dexcom style CGM, heart rate, HRV, steps, sleep stages, SpO2, wrist temperature, insulin and carb logs", "wearables/"],
        ["Fuse both streams", "5 minute timeline; EHR joined only from labs dated before each timestamp", "ingestion/align.py"],
        ["Predict an adverse event", "P(glucose above 180) and P(below 70) within 2 h, plus 1 h and 2 h forecasts", "models/events.py"],
        ["Conceptual doctor dashboard", "Working Streamlit app: triage, patient view, live monitor, results", "dashboard/app.py"],
        ["Synthetic or open data only", "No real patient data; MIMIC IV not used", "CONTRACT.md"], ["Architecture diagram and presentation (PDF or PPT)", "Generated from code so numbers stay in sync", "docs/"], ["Open source licence details", "MIT licence; Synthea is Apache 2.0", "LICENSE"], ["Demo video", "Unlisted YouTube link is in the README", "README"]], fs=10, pad=0.17, until=0.75)
    panel(ax, 9.0, 0.5, 4.1, 6.2, "Submission checklist", LG)
    for i, (t, ok) in enumerate([("Public GitHub repository", True), ("README: team, title, problem, stack, license", True), ("Open source license (MIT)", True), ("Architecture diagram, PDF", True), ("Presentation, PDF", True), ("Synthetic data only", True), ("Demo video on YouTube (link in README)", True), ("Folder name TeamTwins_IGDTUW (stated in README)", True)]):
        y = 6.0 - i * 0.62
        ax.add_patch(Circle((9.35, y), 0.17, fc=GREEN if ok else ORANGE, ec="white", zorder=4)); txt(ax, 9.35, y, "✓" if ok else "!", 10, "white", "bold", "center")
        para(ax, 9.65, y + 0.03, t, 3.3, 9.3, INK, "bold" if not ok else "normal")
    txt(ax, 9.2, 0.85, "All submission items are ready.", 9, ORANGE, "bold")
    F.append(fig)

    # 5 data static
    fig, ax = base(2)
    panel(ax, 0.2, 3.55, 6.4, 3.2, "Synthetic EHR (static stream)")
    for i, (t, b) in enumerate([("patients", "age, sex, BMI, HbA1c, fasting glucose and insulin, HOMA IR, lipids, eGFR, ALT, status, smoking, family history"), ("conditions", "T2D, prediabetes, hypertension, dyslipidemia, obesity, CKD 1 to 3, retinopathy, neuropathy, CAD"), ("labs", "4 to 8 visits in the 730 days before 2026-09-30; 10 lab tests; eGFR by CKD EPI 2021"), ("medications", "metformin, insulin basal and bolus, SGLT2, DPP 4, sulfonylurea, statin, ACE or ARB"), ("genomics", "5 SNPs, risk allele counts, polygenic score; labelled SIMULATED")]):
        tile(ax, 0.4 + (i % 3) * 2.05, 5.1 if i < 3 else 3.7, 1.95, 1.3, t + ".csv", b, [NAVY, CYAN][i % 2], 7.6)
    panel(ax, 6.8, 3.55, 6.3, 3.2, "Cohort mix", LG)
    donut(ax, 8.4, 5.1, 0.9, [("T2D", 20), ("Pre", 15), ("None", 15)], [NAVY, CYAN, GREEN])
    txt(ax, 8.4, 3.68, "Sample profile: 50 patients (T2D, prediabetes, none)", 8, GREY, "bold", "center")
    txt(ax, 9.9, 6.1, "Full profile: 800 patients", 9.5, INK, "bold")
    pill(ax, 9.9, 5.65, 1.5, "type 2 diabetes", 38); pill(ax, 9.9, 5.2, 1.5, "prediabetes", 31, CYAN); pill(ax, 9.9, 4.75, 1.5, "no diabetes", 31, GREEN)
    para(ax, 9.9, 4.2, "Sample: mean age 58.7 (37 to 78), BMI 30.0, HbA1c 6.6; 27 F, 23 M.", 2.8, 8.5, GREY)
    panel(ax, 0.2, 0.5, 4.3, 2.85, "Conditions (sample, n=50)")
    hbars(ax, 0.4, 2.8, 4.0, [("Prediabetes", 35), ("Hypertension", 29), ("Obesity", 24), ("Type 2 diabetes", 20), ("Dyslipidemia", 18), ("Coronary artery", 14), ("CKD stage 3", 2)], 35, NAVY, 0.31, 8.5, 1.5)
    panel(ax, 4.7, 0.5, 4.3, 2.85, "Medicines (sample, patients)", LG)
    hbars(ax, 4.9, 2.8, 4.0, [("Statin", 31), ("ACE inhibitor/ARB", 31), ("Metformin", 24), ("Insulin basal", 10), ("Insulin bolus", 7), ("SGLT2 inhibitor", 6), ("Sulfonylurea", 5), ("DPP 4 inhibitor", 3)], 31, CYAN, 0.28, 8.3, 1.65)
    panel(ax, 9.2, 0.5, 3.9, 2.85, "Genetic markers (simulated)")
    table(ax, 9.35, 2.95, [1.1, 0.95, 0.75, 0.8], ["SNP", "Gene", "RAF", "OR"], [["rs7903146", "TCF7L2", "0.30", "1.40"], ["rs5219", "KCNJ11", "0.36", "1.14"], ["rs1801282", "PPARG", "0.85", "1.14"], ["rs9939609", "FTO", "0.40", "1.20"], ["rs13266634", "SLC30A8", "0.65", "1.12"]], fs=8.3, pad=0.05, until=0.65)
    F.append(fig)

    # 6 wearables
    fig, ax = base(2)
    panel(ax, 0.2, 2.5, 7.2, 4.25, "Wearable streams at native rates")
    table(ax, 0.4, 6.4, [1.6, 1.7, 3.7], ["Device (simulated)", "Metric", "Rate and realism"], [
        ["Dexcom G7 style CGM", "glucose", "5 min, AR(1) noise, 30 min warm up, day 10 sensor change, 15 to 120 min dropouts"],
        ["Apple Watch style", "heart rate", "1 min near activity, else 5 to 10 min"], ["", "HRV, resting HR", "3 to 5 day spot reads, 4 to 8 overnight; 1 resting a day"],
        ["", "steps, energy", "irregular segments up to 26 min"], ["", "sleep stage", "in bed, awake, core, deep, REM"], ["", "SpO2, wrist temp", "overnight; 1 per night"],
        ["Manual log, insulin pen", "carbs, bolus, basal", "meals logged with p = 0.35 to 0.95; boluses 0.5 to 0.97; about 25% carb error"]], fs=8.6, pad=0.05, until=2.65)
    panel(ax, 7.6, 2.5, 5.5, 4.25, "EHR to physiology link", LG)
    pairs = [("HOMA IR, HbA1c, smoking, polygenic score, age", "insulin sensitivity falls"), ("BMI", "glucose clearance falls"), ("eGFR", "insulin clears slower"), ("HbA1c, years since dx, DPP 4", "beta cell response"), ("SGLT2 inhibitor", "lower glucose peaks"), ("Sulfonylurea", "higher hypo risk"), ("Metformin", "hepatic output minus 8%")]
    for i, (a, b) in enumerate(pairs):
        y = 6.2 - i * 0.52
        rbox(ax, 7.75, y - 0.19, 2.55, 0.38, "white", NAVY, 0.06, 1, z=3); txt(ax, 7.82, y, "\n".join(wrap(a, 26)[:2]), 7.8 if len(a) > 26 else 9, INK, "bold", z=6, linespacing=0.95)
        arrow(ax, 10.38, y, 10.78, y); rbox(ax, 10.85, y - 0.19, 2.1, 0.38, NAVY, "none", 0.06, z=3); txt(ax, 10.93, y, b, 9, "white", "bold", z=6)
    panel(ax, 0.2, 0.5, 12.9, 1.8, "Built to be messy, like real devices", LC)
    for i, (n, l) in enumerate([("15-120", "min CGM dropouts"), ("0.35-0.95", "meal log probability"), ("~25%", "carb mis estimate"), ("1 min", "truth, never an input")]):
        circle_stat(ax, 0.45 + i * 3.2, 1.2, n, l, w=2.3, fs=8.5)
    F.append(fig)

    # 7 pipeline
    fig, ax = base(3)
    panel(ax, 0.2, 4.5, 12.9, 2.25, "Pipeline: five stages, each with a test")
    for i, (t, b) in enumerate([("1 Generate", "Synthea EHR, labs, medicines, SNPs; EHR conditioned CGM and watch simulation"), ("2 Align", "raw streams to 5 min grid; interpolate gaps of 15 min or less; keep gap flags"), ("3 Fuse", "join EHR per day from labs, conditions, medicines dated before that day"), ("4 Predict", "LightGBM classifiers and quantile forecasters; patient grouped 5 fold CV"), ("5 Explain, triage", "SHAP, what if, alert log, ranked panel in Streamlit")]):
        x = 0.4 + i * 2.55
        pts = [(x, 4.75), (x + 2.2, 4.75), (x + 2.45, 5.5), (x + 2.2, 6.25), (x, 6.25), (x + 0.25, 5.5)]
        ax.add_patch(Polygon(pts, closed=True, fc=[NAVY, CYAN, NAVY, CYAN, GREEN][i], ec="white", lw=2, zorder=3))
        txt(ax, x + 1.25, 6.0, t, 10.5, "white", "bold", "center")
        for k, ln in enumerate(wrap(b, 25)[:5]): txt(ax, x + 1.25, 5.68 - k * 0.2, ln, 8.8, "white", ha="center")
    panel(ax, 0.2, 0.5, 8.2, 3.8, "Stage, module and key setting", "white")
    table(ax, 0.4, 3.9, [1.4, 3.3, 3.3], ["Stage", "Module", "Key setting"], [["Generate", "ehr/build_ehr.py, wearables/simulate.py", "800 patients x 14 days (full profile)"], ["Align", "ingestion/align.py", "5 min grid; gaps of 15 min or less filled"], ["Fuse", "ingestion/align.py: fuse_with_ehr", "labs dated before the day only"], ["Predict", "models/events.py, features.py", "500 trees, 15 leaves, patient grouped CV"], ["Explain", "dashboard/app.py", "triage, patient, live monitor, results"]], fs=9, pad=0.06, until=0.7)
    panel(ax, 8.6, 0.5, 4.5, 3.8, "Integrity controls", LG)
    for i, (t, b) in enumerate([("Lab join", "only labs dated before each day"), ("Label window", "t+5 to t+120 min only"), ("Patient grouped CV", "no patient in train and test"), ("Streaming = batch", "score_stream equals out of fold"), ("Hidden truth", "1 min glucose never a feature"), ("Disclosed bias", "thresholds on pooled scores")]):
        y = 3.75 - i * 0.55
        ax.add_patch(Circle((8.9, y), 0.17, fc=GREEN, ec="white", zorder=4)); txt(ax, 8.9, y, "✓", 9, "white", "bold", "center")
        txt(ax, 9.2, y + 0.1, t, 10.5, INK, "bold"); txt(ax, 9.2, y - 0.13, b, 9.3, GREY)
    F.append(fig)

    # arch slide
    fig, ax = base(3)
    arch_detail(fig.add_axes([0.01, 0.045, 0.98, 0.9]))
    F.append(fig)

    # 8 models
    fig, ax = base(4)
    panel(ax, 0.2, 3.75, 12.9, 3.0, "Feature groups (all strictly causal)")
    groups = [("Glucose dynamics", "lags 15/30/60 min, rates of change, acceleration, 1 h and 3 h min/max/mean/SD, smoothed slope, minutes to reach 70 or 180"), ("Insulin and carbs", "insulin on board (4 h half life), carbs on board (1 h), ratio, minutes since meal and bolus"), ("Personal history", "lows and highs over 24 h and 72 h, CV, time below 90 or above 160, time since last low, same hour previous 3 days"), ("Physiology", "HR excess, steps, energy, sleep, resting HR trend, SpO2 minimum, wrist temp, time of day, gap flags"), ("Health record", "age, BMI, HbA1c, lipids, eGFR, ALT, 2 year slopes, HOMA IR, comorbidities, medicine flags, SNPs, smoking, family history")]
    for i, (t, b) in enumerate(groups):
        tile(ax, 0.4 + i * 2.55, 3.95, 2.4, 2.45, t, b, [NAVY, CYAN, NAVY, CYAN, GREEN][i], 8.5)
    panel(ax, 0.2, 0.5, 7.6, 3.05, "Models and training", LG)
    table(ax, 0.4, 3.2, [1.7, 5.6], ["Component", "Specification"], [["Classifiers", "LightGBM binary; 500 trees, lr 0.03, 15 leaves, min 300 per leaf, subsample 0.8, colsample 0.5, L2 10"], ["Forecasters", "LightGBM quantile (10, 50, 90%) on change in glucose at +60 and +120 min"], ["Forecast band", "conformalised to 80% coverage on held out patients"], ["Baselines", "trend rule, wearable only model, persistence"], ["Alert rule", "above threshold now and 10 min earlier"], ["Explanations", "per fold boosters kept for SHAP"]], fs=8.4, pad=0.05, until=0.65)
    panel(ax, 8.0, 0.5, 5.1, 3.05, "Why heavily regularised")
    circle_stat(ax, 8.25, 2.8, "0.03", "fused AP lost before tuning", w=4.0, fs=11)
    fitpara(ax, 8.3, 2.15, 1.25, 4.6, "EHR columns are constant per patient, so looser trees memorise training patients. Shallow trees, 300 sample leaves and L2 10 fixed that on held out patients.", INK, "normal", 13)
    para(ax, 8.3, 1.0, "Patient grouped 5 fold CV: no patient is in both train and test folds.", 4.6, 9.3, GREY, "bold")
    F.append(fig)

    # B top features (real)
    import json as _j
    ab = _j.load(open(ROOT / "data/sample/results/ablation.json")); ev = _j.load(open(ROOT / "data/sample/results/events.json"))
    fig, ax = base(4)
    for k, (key, ttl, col, x0) in enumerate([("Wearable-only", "Top features: wearable only model", CYAN, 0.2), ("Fully fused", "Top features: fully fused model", NAVY, 6.75)]):
        panel(ax, x0, 2.3, 6.35, 4.45, ttl, LG if k == 0 else LB)
        items = sorted(ab["top_features"][key].items(), key=lambda kv: -kv[1])[:10]
        mx = items[0][1]
        hbars(ax, x0 + 0.2, 6.1, 6.0, [(n.replace("_", " "), v) for n, v in items], mx, col, 0.375, 9.3, 2.5, "{:,.0f}")
    panel(ax, 0.2, 0.5, 12.9, 1.6, "Reading the importance charts")
    circle_stat(ax, 0.45, 1.25, str(len(ev["features"])), "input features in the event model", w=3.4, fs=13)
    para(ax, 4.6, 1.65, "Minutes since the last logged meal, recent glucose and time of day lead for both models. In the fused model HbA1c and fasting glucose enter the top 10, which is the record acting as a prior. Importances are LightGBM split gain for the 60 minute forecast ablation (sample profile).", 8.3, 10, INK)
    F.append(fig)

    # 9 results
    fig, ax = base(5)
    for i, (n, l) in enumerate([("800", "patients, 14 days each"), ("5 fold", "CV grouped by patient"), ("1.2 M", "held out test moments"), ("2.7%", "hypo base rate")]):
        circle_stat(ax, 0.25 + i * 3.25, 6.35, n, l, w=2.45, fs=11 if len(n) > 4 else 14)
    panel(ax, 0.2, 0.5, 6.4, 5.2, "AUC against a trend rule")
    a = chart(fig, 0.8, 1.0, 5.4, 3.9)
    for i, (n, v, c) in enumerate([("Trend rule", [0.748, 0.842], "#9aa6c0"), ("Wearable only", [0.932, 0.956], CYAN), ("Fused", [0.933, 0.956], NAVY)]):
        a.bar([k + (i - 1) * 0.27 for k in range(2)], v, 0.26, color=c, label=n)
        for k, val in enumerate(v): a.text(k + (i - 1) * 0.27, val + 0.006, f"{val:.3f}", ha="center", fontsize=8, weight="bold")
    a.set_xticks(range(2)); a.set_xticklabels(["Glucose above 180", "Glucose below 70"]); a.set_ylim(0.6, 1.08); a.set_ylabel("AUC"); a.legend(frameon=False, ncol=3, loc="upper center", fontsize=8.5)
    panel(ax, 6.8, 0.5, 6.3, 5.2, "Held out results, balanced setting", LG)
    table(ax, 6.95, 5.3, [0.7, 1.35, 0.6, 0.65, 0.7, 0.65, 0.7], ["Event", "Model", "AUC", "AP", "Ep. prec.", "Caught", "Warn"], [
        [">180", "Trend rule", "0.748", "0.31", "0.30", "93.7%", "40 min"], [">180", "Wearable only", "0.932", "0.66", "0.50", "88.6%", "80 min"], [">180", "Fused", "0.933", "0.67", "0.49", "88.9%", "80 min"],
        ["<70", "Trend rule", "0.842", "0.16", "0.19", "77.0%", "40 min"], ["<70", "Wearable only", "0.956", "0.47", "0.38", "70.6%", "70 min"], ["<70", "Fused", "0.956", "0.47", "0.35", "73.7%", "70 min"]], fs=8.4, pad=0.07, hl=[2, 5], until=3.05)
    para(ax, 6.95, 2.95, "Episode precision: share of alert episodes followed by an event within 2 h. Caught: alert in the 2 h before glucose first left range after an hour in range. Low glucose AP is 18 times the base rate; richer features lifted it from 0.32 to 0.47 on identical labels, 0.50 now.", 5.9, 8.8, GREY)
    banner(ax, 6.95, 0.65, 6.0, 0.95, "HEADLINE", [("0.93 / 0.96", "AUC high / low"), ("80 / 70 min", "median warning"), ("1.85 / 0.55", "alerts per patient day")])
    F.append(fig)

    # 10 operating points + forecasts
    fig, ax = base(5)
    panel(ax, 0.2, 2.9, 6.4, 3.85, "Alert settings: the clinic picks the trade off")
    table(ax, 0.4, 6.35, [0.7, 1.0, 0.9, 1.15, 1.15, 1.2], ["Event", "Setting", "Caught", "Ep. precision", "Row precision", "Alerts / day"], [[">180", "sensitive", "97.1%", "0.33", "0.44", "2.3"], [">180", "balanced", "88.9%", "0.49", "0.59", "1.8"], [">180", "precise", "85.3%", "0.54", "0.64", "1.7"], ["<70", "sensitive", "95.6%", "0.13", "0.21", "1.3"], ["<70", "balanced", "73.7%", "0.35", "0.47", "0.55"], ["<70", "precise", "52.0%", "0.55", "0.66", "0.25"]], fs=8.6, pad=0.06, hl=[1, 4], until=3.65)
    para(ax, 0.4, 3.35, "Precise low setting: 0.25 alerts a day, 55% real, misses about 48% of lows. Sensitive: catches 96% at 13% episode precision.", 6.0, 8.8, GREY)
    panel(ax, 6.8, 2.9, 6.3, 3.85, "Trade off curve", LG)
    a = chart(fig, 7.5, 3.4, 5.2, 2.7)
    for lab, c, v in [("above 180", NAVY, [(97.1, .33), (88.9, .49), (85.3, .54)]), ("below 70", ORANGE, [(95.6, .13), (73.7, .35), (52.0, .55)])]:
        a.plot([q[0] for q in v], [q[1] for q in v], "-o", color=c, lw=2.2, ms=7, label=lab)
        for nm, q in zip(["sens.", "bal.", "prec."], v): a.annotate(nm, q, textcoords="offset points", xytext=(5, 5), fontsize=7.5, color=c)
    a.set_xlabel("Events caught (%)"); a.set_ylabel("Episode precision"); a.legend(frameon=False, fontsize=8.5)
    panel(ax, 0.2, 0.5, 6.4, 2.1, "Forecast error, median mg/dL")
    a = chart(fig, 0.9, 0.75, 2.9, 1.3)
    for i, (n, v, c) in enumerate([("Persistence", [17.4, 26.4], "#9aa6c0"), ("Fused", [11.9, 16.1], NAVY)]):
        a.bar([k + (i - .5) * .36 for k in range(2)], v, .34, color=c, label=n)
        for k, val in enumerate(v): a.text(k + (i - .5) * .36, val + .6, f"{val}", ha="center", fontsize=8, weight="bold")
    a.set_xticks(range(2)); a.set_xticklabels(["+1 h", "+2 h"]); a.set_ylim(0, 33); a.legend(frameon=False, fontsize=7.5, loc="upper left")
    gauge(ax, 4.7, 1.55, 0.5, 80, "80% band coverage", NAVY, 10); txt(ax, 5.75, 1.55, "32% and 39%\nbetter than\npersistence", 9, INK, "bold")
    panel(ax, 6.8, 0.5, 6.3, 2.1, "Reading the results", LG)
    fitpara(ax, 7.0, 2.1, 0.7, 5.9, "The band is conformalised on held out patients, so 80% coverage is measured, not assumed. Median error is 11.9 mg/dL at 1 h and 16.1 at 2 h, against 17.4 and 26.4 for repeating the last reading.", INK, "normal", 14)
    F.append(fig)

    # 11 ablation
    fig, ax = base(5)
    panel(ax, 0.2, 3.2, 12.9, 3.55, "Does the health record help? An honest answer")
    for i, (t, b, c) in enumerate([("After one day of CGM: no", "Personal history features carry what the record knew. Fused and wearable only models tie: AUC 0.933 and 0.956.", NAVY), ("First 24 hours: yes for highs", "AUC 0.916 against 0.905, AP 0.62 against 0.59. For lows AP 0.38 against 0.40, within noise.", GREEN), ("Against no history model", "The record cuts 60 minute error from 13.5 to 12.8 mg/dL.", CYAN), ("What this really shows", "The simulator is driven by the EHR, so this shows the pipeline recovers that link, not that it exists in real patients.", ORANGE)]):
        tile(ax, 0.4 + i * 3.2, 3.4, 3.05, 2.85, t, b, c, 9.8)
    panel(ax, 0.2, 0.5, 6.4, 2.5, "60 minute forecast MAE, sample (50 patients)", LG)
    a = chart(fig, 2.0, 0.8, 4.3, 1.7)
    names = ["Persistence", "Wearable only", "Fused (snapshot)", "Fully fused"]; vals = [16.89, 14.46, 13.89, 13.92]
    a.barh(names[::-1], vals[::-1], color=["#9aa6c0", CYAN, NAVY, NAVY][::-1])
    for i, v in enumerate(vals[::-1]): a.text(v + 0.1, i, f"{v}", va="center", fontsize=8.5, weight="bold")
    a.set_xlim(10, 18)
    panel(ax, 6.8, 0.5, 6.3, 2.5, "Paired Wilcoxon tests across patients")
    circle_stat(ax, 7.0, 2.3, "0.57", "mg/dL lower MAE with snapshot EHR", w=5.1, fs=11)
    circle_stat(ax, 7.0, 1.6, "70%", "of patients improve, p = 0.0002", w=5.1, fs=11)
    circle_stat(ax, 7.0, 0.9, "p 0.57", "full EHR adds nothing on top", w=5.1, fs=10)
    F.append(fig)

    # C committed sample results (real)
    ec = pd.read_csv(ROOT / "data/sample/results/events.csv"); fc_ = ev["forecast"]; cs_ = ev["cold_start"]
    fig, ax = base(5)
    for i, (n, l) in enumerate([("50", "patients (sample profile)"), (f"{ev['patient_days']:.0f}", "patient days"), (f"{ev['n_rows']:,}", "scored moments"), ("326 / 41", "high / low onsets")]):
        circle_stat(ax, 0.25 + i * 3.25, 6.4, n, l, w=2.45, fs=9.5 if len(n) > 4 else 13)
    panel(ax, 0.2, 2.35, 7.6, 3.55, "Committed results, sample profile (balanced setting)")
    rows = []
    for _, r in ec[ec.event != "any"].iterrows():
        rows.append([">180" if r.event == "hyper" else "<70", r.model, f"{r.AUC:.3f}", f"{r.AP:.2f}", f"{r.episode_precision:.2f}", f"{r.onsets_caught_pct:.1f}%", f"{r.median_lead_min:.0f} min", f"{r.alerts_per_patient_day:.2f}"])
    table(ax, 0.4, 5.55, [0.65, 1.55, 0.8, 0.7, 0.9, 0.95, 0.95, 0.9], ["Event", "Model", "AUC", "AP", "Ep. prec.", "Caught", "Warn", "Alerts/day"], rows, fs=8.5, pad=0.06, hl=[4, 5], until=2.5)
    panel(ax, 8.0, 2.35, 5.1, 3.55, "First 24 hours (cold start)", LG)
    table(ax, 8.2, 5.55, [1.5, 0.9, 0.8, 0.8], ["Model", "Event", "AUC", "AP"], [["Wearable only", ">180", f"{cs_['Wearable-only|hyper']['AUC']:.3f}", f"{cs_['Wearable-only|hyper']['AP']:.2f}"], ["Fused", ">180", f"{cs_['Fused|hyper']['AUC']:.3f}", f"{cs_['Fused|hyper']['AP']:.2f}"], ["Wearable only", "<70", f"{cs_['Wearable-only|hypo']['AUC']:.3f}", f"{cs_['Wearable-only|hypo']['AP']:.2f}"], ["Fused", "<70", f"{cs_['Fused|hypo']['AUC']:.3f}", f"{cs_['Fused|hypo']['AP']:.2f}"]], fs=9, pad=0.06, until=3.1)
    para(ax, 8.2, 2.85, f"{cs_['Fused|hyper']['n']:,} cold start moments per model.", 4.7, 9, GREY, "bold")
    panel(ax, 0.2, 0.5, 12.9, 1.65, "Forecast and band on the sample profile", LC)
    for i, (n, l) in enumerate([(f"{fc_['60min']['median_MAE']:.1f}", f"mg/dL error at 1 h (persistence {fc_['60min']['persistence_MAE']:.1f})"), (f"{fc_['120min']['median_MAE']:.1f}", f"mg/dL error at 2 h (persistence {fc_['120min']['persistence_MAE']:.1f})"), (f"{fc_['60min']['empirical_coverage']:.0%}", f"band coverage, widened {fc_['60min']['widen_mg_dl']:.1f} mg/dL at 1 h")]):
        circle_stat(ax, 0.45 + i * 4.25, 1.2, n, l, w=3.4, fs=11)
    para(ax, 0.45, 0.7, "The small cohort gives lower AUC than the 800 patient run; thresholds maximise F1 on pooled out of fold scores. Numbers here regenerate from data/sample/results.", 12.2, 8.8, GREY)
    F.append(fig)

    # 12 dashboard (real data)
    fig, ax = base(6)
    panel(ax, 0.2, 2.6, 5.6, 4.15, "Triage board (real sample output)")
    txt(ax, 0.4, 6.15, "Patient", 9, GREY, "bold"); txt(ax, 2.0, 6.15, "P(glucose above 180 within 2 h)", 9, GREY, "bold")
    for i, rw in top.iterrows():
        y = 5.8 - i * 0.37
        txt(ax, 0.4, y, str(rw.patient_id)[:8], 9, INK, "bold")
        rbox(ax, 2.0, y - 0.1, 3.0, 0.2, "#e1e4ea", "none", 0.08, z=2); rbox(ax, 2.0, y - 0.1, max(3.0 * rw.p_hyper, 0.05), 0.2, ORANGE if rw.p_hyper > 0.4 else NAVY, "none", 0.08, z=3)
        txt(ax, 5.1, y, f"{rw.p_hyper:.0%}", 9, INK, "bold")
    panel(ax, 6.0, 2.6, 7.1, 4.15, f"Patient view (real traces, patient {pid})", LG)
    a = chart(fig, 6.6, 3.0, 6.2, 2.8)
    t = (d.timestamp - d.timestamp.iloc[0]).dt.total_seconds() / 3600
    a.plot(t, d.glucose, color=NAVY, lw=1.6, label="glucose")
    a.axhline(180, color=ORANGE, ls="--", lw=1); a.axhline(70, color="#c0392b", ls="--", lw=1)
    a.fill_between(t, d.fc_lo, d.fc_hi, color=CYAN, alpha=.25, label="1 h forecast band")
    b = a.twinx(); b.plot(t, d.p_hyper, color=GREEN, lw=1.4, label="P(>180)"); b.set_ylim(0, 1); b.tick_params(labelsize=8); b.spines[["top"]].set_visible(False)
    a.set_xlabel("hours"); a.set_ylabel("mg/dL"); a.legend(frameon=False, fontsize=8, loc="upper left")
    for i, (t_, b_, c) in enumerate([("Why flagged", "SHAP drivers from the fold model that never saw the patient, each bar tagged wearable or health record", NAVY), ("What if", "toggle a medicine or activity and the risk is recomputed; shown as association, not causal advice", CYAN), ("Live monitor", "replays a patient tick by tick and scores only the data received so far, with no look ahead", NAVY), ("Alert log", "each alert is marked confirmed or false once 2 hours have passed", CYAN)]):
        tile(ax, 0.2 + i * 3.25, 0.5, 3.1, 1.95, t_, b_, c, 9.5)
    F.append(fig)

    # 13 realtime + tests
    fig, ax = base(6)
    panel(ax, 0.2, 4.4, 12.9, 2.35, "Real time path: streaming equals batch")
    for i, t_ in enumerate(["New readings arrive", "Rebuild features from history so far", "Score with held out fold model", "Risk, band, SHAP", "Alert needs 2 reads above threshold"]):
        x = 0.45 + i * 2.55; rbox(ax, x, 5.0, 2.2, 1.0, [NAVY, CYAN, NAVY, CYAN, GREEN][i], "none", 0.1, z=3); para(ax, x + 0.12, 5.75, t_, 2.0, 11, "white", "bold")
        if i < 4: arrow(ax, x + 2.22, 5.5, x + 2.53, 5.5)
    txt(ax, 0.45, 4.7, "tests::test_stream_features_equal_batch_and_scores_match_oof proves no look ahead.", 9.5, GREY, "bold")
    panel(ax, 0.2, 0.5, 8.4, 3.7, "Tests that guard the claims", LG)
    table(ax, 0.4, 3.85, [3.9, 4.2], ["Test", "Checks"], [["bmi_consistent, homa_ir_derived", "derived fields are consistent"], ["latest_labs_match, lab_visits_4_to_8", "latest visit equals patient table"], ["meds_match_conditions, complications_need_diabetes", "clinical rules hold"], ["genomics_complete_and_labeled", "5 SNPs each, labelled SIMULATED"], ["raw_schema_and_native_rates, small_gaps_only", "native rates, only small gaps filled"], ["fuse_no_future_lab_leakage", "no future lab reaches a timestamp"], ["event_label_uses_only_future_window, onsets_and_lead_time", "label window and lead time"], ["stream_features_equal_batch", "live equals batch"]], fs=8.2, pad=0.05, until=0.65)
    panel(ax, 8.8, 0.5, 4.3, 3.7, "Reproduce")
    for i, (n, l) in enumerate([("sample", "50 patients x 7 days, minutes"), ("full", "800 x 14 days, ~1 h on 10 cores"), ("py 3.12", "Java 17 for Synthea, libomp")]):
        circle_stat(ax, 9.0, 3.4 - i * 0.85, n, l, w=3.4, fs=8.5)
    txt(ax, 9.0, 0.95, "python run_pipeline.py --profile sample", 8, INK, family="monospace"); txt(ax, 9.0, 0.7, "streamlit run dashboard/app.py", 8, INK, family="monospace")
    F.append(fig)

    # 14 impact
    fig, ax = base(7)
    panel(ax, 0.2, 0.5, 6.6, 6.25, "A tiered design a clinic can run")
    for i, (w_, t_, b_, c) in enumerate([(6.0, "Wearable and record data", "800 patients, every 5 minutes", NAVY), (5.0, "Model alerts patients first", "about 2.4 alerts a day each at balanced", CYAN), (3.9, "Triage board", "only patients in alert now", NAVY), (2.8, "Clinician acts", "who, when and why on one screen", GREEN)]):
        x = 0.5 + (6.0 - w_) / 2; y = 5.2 - i * 1.15
        ax.add_patch(Polygon([(x, y + 0.95), (x + w_, y + 0.95), (x + w_ - 0.3, y), (x + 0.3, y)], fc=c, ec="white", lw=2, zorder=3))
        txt(ax, 3.5, y + 0.62, t_, 12, "white", "bold", "center"); txt(ax, 3.5, y + 0.3, b_, 10, "white", ha="center")
    para(ax, 0.5, 0.95, "The patient is alerted first; raw alert volume never reaches the doctor.", 6.0, 9.3, GREY, "bold")
    panel(ax, 7.0, 3.1, 6.1, 3.65, "Alert volume per patient day", LG)
    pill(ax, 7.3, 6.0, 3.0, "high alerts 1.85", 46, NAVY); pill(ax, 7.3, 5.5, 3.0, "low alerts 0.55", 14, CYAN); pill(ax, 7.3, 5.0, 3.0, "trend rule highs 3.9", 100, "#9aa6c0")
    gauge(ax, 8.2, 3.95, 0.5, 49, "high alerts real", NAVY, 10); gauge(ax, 10.2, 3.95, 0.5, 35, "low alerts real", CYAN, 10); gauge(ax, 12.0, 3.95, 0.5, 55, "precise low real", GREEN, 10)
    panel(ax, 7.0, 0.5, 6.1, 2.35, "Relevance to India")
    fitpara(ax, 7.2, 2.3, 0.65, 5.7, "Few clinicians serve many patients, and the risk sits between visits. A small team can supervise a large panel, and each clinic sets the sensitive, balanced or precise dial to its own staffing and risk tolerance.", INK, "normal", 14)
    F.append(fig)

    # 15 limitations + roadmap
    fig, ax = base(8)
    panel(ax, 0.2, 0.5, 6.4, 6.25, "Limitations, stated plainly", LC)
    for i, (t_, b_) in enumerate([("Simulator is EHR driven", "fusion gain shows recovery of that link, not proof it exists in real patients"), ("Hypoglycaemia is weaker", "about 65% of low alert episodes are false at balanced; unlogged insulin and meals remain"), ("Not a validated T2D model", "type 1 UVA/Padova ODE with PI controller used as a generic glucose insulin system"), ("Optimistic thresholds", "chosen on pooled held out scores; not tuned to a clinical cost"), ("Associations, not causes", "what if answers are not treatment advice"), ("Simulated inputs", "genotypes, polygenic score, devices; research code, not a medical device")]):
        y = 6.2 - i * 0.95
        ax.add_patch(Circle((0.55, y), 0.2, fc=ORANGE, ec="white", zorder=4)); txt(ax, 0.55, y, str(i + 1), 9.5, "white", "bold", "center")
        txt(ax, 0.9, y + 0.14, t_, 11.5, INK, "bold"); fitpara(ax, 0.9, y - 0.12, y - 0.72, 5.5, b_, GREY, "normal", 11.5)
    panel(ax, 6.8, 0.5, 6.3, 6.25, "Roadmap ahead")
    for i, (t_, b_, c) in enumerate([("Validate", "same ingestion adapter on consented real data: MIMIC IV for the record, an open CGM dataset for sensors", NAVY), ("Calibrate", "tune thresholds to a clinical cost of missed versus false alarms with clinicians", CYAN), ("Connect", "real device adapter for the live monitor, per patient online recalibration", NAVY), ("Extend", "reuse ingestion and fusion for hypertension or cardiovascular risk", GREEN)]):
        y = 5.2 - i * 1.4
        rbox(ax, 7.05, y, 1.5, 1.1, c, "none", 0.1, z=3); txt(ax, 7.8, y + 0.55, t_, 11, "white", "bold", "center")
        rbox(ax, 8.65, y, 4.25, 1.1, "white", c, 0.1, 1.4, z=2); fitpara(ax, 8.8, y + 0.85, y + 0.1, 4.0, b_, INK, "normal", 12.5)
        if i < 3: arrow(ax, 7.8, y - 0.02, 7.8, y - 0.3)
    F.append(fig)

    # D codebase
    fig, ax = base(8)
    panel(ax, 0.2, 0.5, 6.4, 6.25, "Codebase at a glance (lines of Python)")
    files = [("dashboard/app.py", 456), ("models/events.py", 268), ("wearables/simulate.py", 210), ("ehr/build_ehr.py", 208), ("ingestion/align.py", 132), ("tests/test_pipeline.py", 102), ("features.py", 91), ("models/ablation.py", 67), ("wearables/generate.py", 52), ("wearables/derive_params.py", 39)]
    hbars(ax, 0.4, 6.1, 6.0, files, 456, NAVY, 0.45, 9.3, 2.5)
    para(ax, 0.45, 1.3, "About 1,600 lines in total. The simulator, ingestion, models, dashboard and tests are separate modules with a single run_pipeline.py entry point.", 5.9, 9.5, GREY, "bold")
    panel(ax, 6.8, 3.4, 6.3, 3.35, "Pipeline stages and outputs", LG)
    table(ax, 7.0, 6.35, [1.5, 4.5], ["Stage", "Output under data/<profile>/"], [["synthea", "synthea_raw (not committed)"], ["ehr", "ehr/*.csv, sim_params.csv"], ["wearables", "raw_wearables, truth (not committed)"], ["ingest", "timeseries.parquet, fused_timeline.parquet"], ["models", "results/ablation.csv and json"], ["events", "events.csv, json, event_risk.parquet, event_models"]], fs=9, pad=0.05, until=3.6)
    panel(ax, 6.8, 0.5, 6.3, 2.7, "Run it yourself")
    for i, c_ in enumerate(["uv venv --python 3.12 .venv", "python run_pipeline.py --profile sample", "python run_pipeline.py --profile full", "streamlit run dashboard/app.py", "pytest tests"]):
        rbox(ax, 7.0, 2.55 - i * 0.43, 5.9, 0.34, "white", NAVY, 0.06, 1, z=3); txt(ax, 7.12, 2.72 - i * 0.43, c_, 9, INK, family="monospace", z=6)
    F.append(fig)

    # 16 closing
    fig, ax = base(8)
    panel(ax, 0.2, 2.6, 6.6, 4.15, "Stack and licence")
    table(ax, 0.4, 6.35, [1.5, 4.8], ["Layer", "Tools"], [["Synthetic EHR", "Synthea (Apache 2.0, Java 17), custom generators"], ["Wearables", "simglucose (UVA/Padova ODE), behaviour model"], ["Data", "pandas, NumPy, SciPy, pyarrow"], ["Modelling", "LightGBM, scikit-learn, SHAP contributions"], ["Interface", "Streamlit, Plotly"], ["Quality", "pytest, 13 tests"], ["Licence", "MIT; dependencies keep their own"]], fs=8.8, pad=0.06, until=2.75)
    panel(ax, 7.0, 2.6, 6.1, 4.15, "Deliverables", LG)
    for i, (n, l) in enumerate([("README", "team, title, problem, stack, licence"), ("LICENSE", "MIT, public repository"), ("docs/architecture.pdf", "architecture diagram"), ("docs/TeamTwins.pdf", "this deck"), ("Video", "demo on YouTube, link in README")]):
        rbox(ax, 7.2, 6.0 - i * 0.7, 2.1, 0.5, NAVY if i % 2 == 0 else CYAN, "none", 0.08, z=3); txt(ax, 8.25, 6.25 - i * 0.7, n, 8.8, "white", "bold", "center")
        txt(ax, 9.45, 6.25 - i * 0.7, l, 9, INK, "bold")
    rbox(ax, 0.2, 0.5, 12.9, 1.8, NAVY, "none", 0.12, z=2)
    txt(ax, 0.6, 1.65, "Same patient. Earlier warning.", 26, "white", "bold"); txt(ax, 0.6, 1.0, "Thank you from Team TeamTwins: Anisha Gupta (IGDTUW) and Vibhor Sharma (GL Bajaj)", 11, "#b8c7ee")
    for i, t_ in enumerate(["EARLIER WARNING", "BETTER TRIAGE", "HONEST RESULTS", "OPEN CODE"]):
        ax.add_patch(Circle((8.3 + i * 1.3, 1.65), 0.26, fc=GREEN, ec="white", zorder=4)); txt(ax, 8.3 + i * 1.3, 1.65, "✓", 12, "white", "bold", "center")
        txt(ax, 8.3 + i * 1.3, 1.1, t_, 6.5, "white", "bold", "center")
    F.append(fig)
    return F


if __name__ == "__main__":
    pres = build()
    with PdfPages(OUT / "architecture.pdf") as pdf:
        pdf.savefig(architecture())
    with PdfPages(OUT / "TeamTwins.pdf") as pdf:
        for f in pres: pdf.savefig(f)
    plt.close("all")
    print(len(pres), "slides")
