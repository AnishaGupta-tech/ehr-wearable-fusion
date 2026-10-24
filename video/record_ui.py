"""Record the Streamlit dashboard pages for the demo video (needs the app running on :8765)."""
import json, subprocess, sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent / "ui"; OUT.mkdir(exist_ok=True)
URL = "http://localhost:8791"
W, H = 1654, 930
only = set(sys.argv[1:])


def settle(page, ms=2500):
    page.wait_for_timeout(ms)
    try: page.wait_for_selector("[data-testid=stStatusWidget]", state="detached", timeout=20000)
    except Exception: pass


def nav(page, label):
    page.locator(f"[data-testid=stSidebar] label:has-text('{label}')").first.click(); settle(page, 3500)


def smooth_scroll(page, total, steps, pause=60):
    for _ in range(steps):
        page.evaluate(f"""() => {{ const m = document.querySelector('[data-testid=stMain]') || document.scrollingElement; m.scrollBy(0, {total / steps}); }}""")
        page.wait_for_timeout(pause)


def pick_dataset(page, name="full"):
    box = page.locator("[data-testid=stSidebar] [data-testid=stSelectbox]").first
    box.click(); page.wait_for_timeout(500); page.get_by_role("option", name=name).click(); settle(page, 5000)


def record(p, name, fn):
    if only and name not in only: return
    tc = time.time()
    ctx = p.chromium.launch().new_context(viewport={"width": W, "height": H}, record_video_dir=str(OUT / "_raw"), record_video_size={"width": W, "height": H})
    page = ctx.new_page(); page.goto(URL); page.wait_for_selector("[data-testid=stSidebar]", timeout=60000); settle(page, 3500)
    pick_dataset(page)
    t0 = time.time(); off = t0 - tc; page.wait_for_timeout(1)
    fn(page)
    vid = page.video; ctx.close(); path = vid.path()
    Path(path).replace(OUT / f"{name}.webm")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{off + 0.6:.2f}", "-i", str(OUT / f"{name}.webm"), "-vf", "fps=30", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an", str(OUT / f"{name}.mp4")], check=True)
    print(name, "setup", round(off, 1), "s, shown", round(time.time() - t0, 1), "s")


def cohort(page):
    nav(page, "Cohort overview"); page.wait_for_timeout(3000); smooth_scroll(page, 1400, 40, 120); page.wait_for_timeout(1500)


def triage(page):
    nav(page, "Triage board"); page.wait_for_timeout(3000); smooth_scroll(page, 700, 30, 120); page.wait_for_timeout(1500)


def patient(page):
    nav(page, "Patient view"); page.wait_for_timeout(3500); smooth_scroll(page, 1500, 50, 120); page.wait_for_timeout(2000)


def why(page):
    nav(page, "Patient view"); page.wait_for_timeout(3000); smooth_scroll(page, 900, 20, 100); page.wait_for_timeout(2500)
    smooth_scroll(page, 900, 25, 120); page.wait_for_timeout(3000)


def whatif(page):
    nav(page, "Patient view"); page.wait_for_timeout(3000); smooth_scroll(page, 1800, 30, 100); page.wait_for_timeout(1500)
    smooth_scroll(page, 1500, 40, 120); page.wait_for_timeout(3000)


def tabs(page):
    nav(page, "Patient view"); page.wait_for_timeout(2500)
    for t in ["Health record", "Genetics (simulated)", "Devices"]:
        page.get_by_role("tab", name=t).click(); settle(page, 2000); smooth_scroll(page, 500, 15, 100); page.wait_for_timeout(1800)
        page.evaluate("() => (document.querySelector('[data-testid=stMain]')||document.scrollingElement).scrollTo(0, 0)")


def live(page):
    nav(page, "Live monitor"); page.wait_for_timeout(2500)
    page.get_by_role("button", name="Jump to next alert").click(); settle(page, 3000)
    page.locator("label:has-text('Streaming')").first.click(); page.wait_for_timeout(14000)
    smooth_scroll(page, 700, 20, 150); page.wait_for_timeout(5000)


def results(page):
    nav(page, "Model results"); page.wait_for_timeout(3000); smooth_scroll(page, 3000, 80, 130); page.wait_for_timeout(2000)


with sync_playwright() as p:
    for n, f in [("cohort", cohort), ("triage", triage), ("patient", patient), ("why", why), ("whatif", whatif), ("tabs", tabs), ("live", live), ("results", results)]:
        try: record(p, n, f)
        except Exception as e: print("FAILED", n, repr(e)[:300])
