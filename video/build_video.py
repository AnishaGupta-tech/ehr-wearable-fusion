"""Assemble the TeamTwins demo video from slide frames, dashboard clips and scene audio.

Usage (from repo root):
    python video/build_video.py --draft            # placeholder macOS voice, for timing and review
    python video/build_video.py                    # uses video/audio/scene_XX.wav (the real recording)
    python video/build_video.py --scenes 1 2       # build only some scenes
Outputs: video/clips/*.mp4 per scene, video/out/TeamTwins_demo.mp4 (all scenes joined).
"""
import argparse, math, re, subprocess, sys, textwrap
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FRAMES, UI, AUDIO, CLIPS, TMP, OUT = (HERE / d for d in ("frames", "ui", "audio", "clips", "tmp", "out"))
for d in (AUDIO, CLIPS, TMP, OUT): d.mkdir(exist_ok=True)
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
FONT, FONT_B = str(FONT_DIR / "DejaVuSans.ttf"), str(FONT_DIR / "DejaVuSans-Bold.ttf")
W, H, FPS = 1920, 1080, 30
SW, SH = 1654, 930          # slide area; the caption band sits below it
XOFF = (W - SW) // 2
BAND = (13, 27, 61)
NAVY, GREEN = (21, 52, 124), (37, 192, 109)

SCENES = {
    1: ("Welcome", [1]), 2: ("The problem", [2]), 3: ("Clinical definitions", [3]), 4: ("Organiser and brief", [4, 5]),
    5: ("Data: record and wearables", [6, 7]), 6: ("Pipeline and architecture", [8, 9]), 7: ("Features and models", [10, 11]),
    8: ("Results", [12, 13, 14, 15]), 9: ("Live dashboard demo", [16]), 10: ("Real time path and tests", [17]),
    11: ("Impact, limitations, roadmap", [18, 19]), 12: ("Codebase and close", [20, 21]),
}
# [LIVE: name] markers in scene 9 -> ui clip file
UI_MAP = {"Cohort overview": "cohort", "Triage board": "triage", "Patient view, Glucose and wearables tab": "patient",
          "Why flagged": "why", "What if panel": "whatif", "Health record, Genetics, Devices tabs": "tabs",
          "Live monitor": "live", "Model results": "results"}


def run(cmd, quiet=True):
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode:
        sys.exit(f"FAILED: {' '.join(map(str, cmd))}\n{r.stderr[-1500:]}")
    return r.stdout


def dur(path):
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)]).strip())


def parse_script():
    txt = (HERE / "script.md").read_text()
    out = {}
    for m in re.finditer(r"## Scene (\d+) \|.*?\nVOICEOVER:\n(.*?)(?=\n## Scene|\n---)", txt, re.S):
        out[int(m.group(1))] = m.group(2).strip()
    return out


def split_segments(scene_text):
    """-> list of (ui_key or None, text). Text before the first [LIVE:] marker belongs to the slide."""
    parts = re.split(r"\[LIVE: ([^\]]+)\]\n?", scene_text)
    segs = [(None, parts[0].strip())]
    for i in range(1, len(parts), 2):
        segs.append((UI_MAP[parts[i].strip()], parts[i + 1].strip()))
    return [(k, t) for k, t in segs if t]


def sentences(text):
    out = []
    for para in re.split(r"\n\s*\n", text):
        for s in re.split(r"(?<=[.!?])\s+", para.replace("\n", " ").strip()):
            while len(s) > 125:
                cut = max(s.rfind(", ", 0, 110), s.rfind("; ", 0, 110), s.rfind(": ", 0, 110))
                if cut < 50: cut = s.rfind(" ", 0, 110)
                out.append(s[:cut + 1].strip()); s = s[cut + 1:].strip()
            if s: out.append(s)
    return out


def spoken(text):
    t = re.sub(r"\bA I\b", "AI", text)
    return re.sub(r"\b([A-Z]) ([A-Z0-9]) ?([A-Z0-9])? ?([A-Z0-9])?\b", lambda m: "".join(g for g in m.groups() if g), t) if False else t


def draft_audio(n, text):
    """Placeholder voice: macOS 'say', only for checking timing."""
    plain = re.sub(r"\[LIVE:[^\]]*\]", " ", text)
    aiff, wav = TMP / f"draft_{n:02d}.aiff", AUDIO / f"draft_{n:02d}.wav"
    (TMP / f"draft_{n:02d}.txt").write_text(plain)
    run(["say", "-v", "Samantha", "-r", "150", "-o", str(aiff), "-f", str(TMP / f"draft_{n:02d}.txt")])
    run(["ffmpeg", "-y", "-i", str(aiff), "-ar", "44100", "-ac", "1", str(wav)])
    return wav


def font(path, size): return ImageFont.truetype(path, size)


def overlay_png(path, caption, title=None, progress=0.0, scene_no=None):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rectangle([0, SH, W, H], fill=BAND + (255,))
    if title:
        d.rectangle([60, SH + 14, 66, SH + 40], fill=GREEN + (255,))
        d.text((80, SH + 27), f"{scene_no:02d}  {title.upper()}", font=font(FONT_B, 19), fill=(158, 240, 193, 255), anchor="lm")
        d.text((W - 60, SH + 27), "TeamTwins  |  Glucose Digital Twin", font=font(FONT, 18), fill=(150, 165, 200, 255), anchor="rm")
    if caption:
        f = font(FONT, 36); lines = textwrap.wrap(caption, 92)[:2]; lh = 42
        y0 = SH + 52 + (2 - len(lines)) * lh / 2
        for i, l in enumerate(lines):
            d.text((W / 2, y0 + i * lh + lh / 2), l, font=f, fill=(255, 255, 255, 255), anchor="mm")
    d.rectangle([0, H - 8, W, H], fill=(40, 55, 95, 255)); d.rectangle([0, H - 8, int(W * progress), H], fill=GREEN + (255,))
    im.save(path)


def still_clip(slide_png, overlay, t0, t1, slide_len, out, fade_in, fade_out, pan_seed):
    """Slow pan/zoom drift across the slide while the caption overlay sits still."""
    d = t1 - t0
    sw, sh = int(SW * 1.03) // 2 * 2, int(SH * 1.03) // 2 * 2
    dx, dy = (sw - SW), (sh - SH)
    sx, sy = (0, 0) if pan_seed % 2 == 0 else (dx, 0)
    ex, ey = (dx, dy) if pan_seed % 2 == 0 else (0, dy)
    p0, p1 = t0 / slide_len, t1 / slide_len
    xe = f"({sx}+({ex}-{sx})*({p0}+({p1}-{p0})*t/{d}))"; ye = f"({sy}+({ey}-{sy})*({p0}+({p1}-{p0})*t/{d}))"
    vf = f"[0:v]scale={sw}:{sh},crop={SW}:{SH}:x='{xe}':y='{ye}',pad={W}:{H}:{XOFF}:0:color=0x0d1b3d,format=yuv420p[b];[b][1:v]overlay=0:0,format=yuv420p"
    if fade_in: vf += f",fade=t=in:st=0:d=0.5"
    if fade_out: vf += f",fade=t=out:st={d - 0.5}:d=0.5"
    run(["ffmpeg", "-y", "-loop", "1", "-framerate", str(FPS), "-t", f"{d:.3f}", "-i", str(slide_png), "-loop", "1", "-framerate", str(FPS), "-t", f"{d:.3f}", "-i", str(overlay),
         "-filter_complex", vf, "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(out)])


def ui_clip(src, overlay, off, d, out, fade_in, fade_out):
    vf = f"[0:v]scale={SW}:{SH}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:0:color=0xffffff,setsar=1,format=yuv420p[b];[b][1:v]overlay=0:0,format=yuv420p"
    if fade_in: vf += ",fade=t=in:st=0:d=0.4"
    if fade_out: vf += f",fade=t=out:st={d - 0.4}:d=0.4"
    run(["ffmpeg", "-y", "-ss", f"{off:.3f}", "-t", f"{d:.3f}", "-i", str(src), "-loop", "1", "-framerate", str(FPS), "-t", f"{d:.3f}", "-i", str(overlay),
         "-filter_complex", vf, "-r", str(FPS), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(out)])


def retime(src, target, out):
    """Stretch (or hold the last frame of) a UI clip to exactly `target` seconds."""
    d0 = dur(src)
    if d0 >= target:
        vf = f"setpts={target / d0}*PTS,fps={FPS}"
    else:
        vf = f"fps={FPS},tpad=stop_mode=clone:stop_duration={target - d0 + 0.1}"
    run(["ffmpeg", "-y", "-i", str(src), "-vf", vf, "-t", f"{target:.3f}", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(out)])


def aligned_durations(n, seg_sents, ad):
    """Per-sentence durations from the real word times of the recording (script words aligned to transcript words)."""
    import difflib, json
    wf = TMP / f"scene_{n:02d}_words.json"
    if not wf.exists(): return None
    tw = json.loads(wf.read_text()); nm = lambda w: re.sub(r"[^a-z0-9]", "", w.lower())
    ttok = [nm(w[2]) for w in tw]
    flat = [(si, i, [nm(w) for w in x.split() if nm(w)]) for si, ss in enumerate(seg_sents) for i, x in enumerate(ss)]
    stok, owner = [], []
    for fi, (_, _, toks) in enumerate(flat):
        stok += toks; owner += [fi] * len(toks)
    m = difflib.SequenceMatcher(None, stok, ttok, autojunk=False)
    t_of = {}
    for a, b, size in m.get_matching_blocks():
        for k in range(size): t_of[a + k] = tw[b + k][0]
    starts = [None] * len(flat)
    for ti in range(len(stok)):
        fi = owner[ti]
        if starts[fi] is None and ti in t_of: starts[fi] = t_of[ti]
    # fill gaps by interpolation, force monotonic
    last = 0.0
    for i in range(len(starts)):
        if starts[i] is None:
            nxt = next((starts[j] for j in range(i + 1, len(starts)) if starts[j] is not None), ad - 0.6)
            starts[i] = last + (nxt - last) / 2 if nxt > last else last
        starts[i] = max(starts[i], last); last = starts[i]
    starts[0] = 0.0
    ends = starts[1:] + [ad]
    durs = [max(e - s, 0.4) for s, e in zip(starts, ends)]
    out, k = [], 0
    for ss in seg_sents:
        out.append(durs[k:k + len(ss)]); k += len(ss)
    return out


def build_scene(n, text, audio, total_scenes_dur, start_time):
    title, slides = SCENES[n]
    ad = dur(audio) + 0.6
    segs = split_segments(text)
    seg_sents = [sentences(t) for _, t in segs]
    sent_dur = aligned_durations(n, seg_sents, ad)
    if sent_dur is None:   # no transcript: fall back to character share
        chars = [sum(len(x) for x in ss) for ss in seg_sents]
        seg_durs = [ad * c / sum(chars) for c in chars]
        sent_dur = [[sd * len(x) / max(sum(len(y) for y in ss), 1) for x in ss] for sd, ss in zip(seg_durs, seg_sents)]
    seg_durs = [sum(d) for d in sent_dur]
    parts = []; t_scene = 0.0; k = 0
    for si, ((ui_key, seg_text), sd) in enumerate(zip(segs, seg_durs)):
        sents = seg_sents[si]; weights = sent_dur[si]; tot = sum(weights)
        if ui_key is None:
            # spread this segment over the scene's slides
            per_slide = []
            share = [1.0 / len(slides)] * len(slides) if si == 0 and len(segs) == 1 else None
            acc = 0; bounds = []
            for i, s in enumerate(sents):
                acc += weights[i]; bounds.append(acc / tot)
            slide_of = [min(int(b * len(slides) - 1e-9), len(slides) - 1) for b in bounds]
            # slide of a sentence = slide index of its end position (first sentences land on first slide)
            starts = [0.0]
            for w in weights[:-1]: starts.append(starts[-1] + w)
            slide_of = [min(int((st / tot) * len(slides)), len(slides) - 1) for st in starts]
            src_kind = ("slide", slide_of)
        else:
            src = UI / f"{ui_key}.mp4"
            if not src.exists(): sys.exit(f"missing dashboard clip {src}")
            rt = TMP / f"s{n:02d}_{ui_key}_retimed.mp4"; retime(src, sd, rt)
            src_kind = ("ui", rt)
        t_in_seg = 0.0
        # slide length bookkeeping for pan continuity
        slide_totals = {}
        if src_kind[0] == "slide":
            for i, s in enumerate(sents):
                slide_totals[src_kind[1][i]] = slide_totals.get(src_kind[1][i], 0) + sd * weights[i] / tot
        slide_pos = {j: 0.0 for j in slide_totals}
        for i, s in enumerate(sents):
            cd = sd * weights[i] / tot
            ov = TMP / f"ov_{n:02d}_{k:03d}.png"
            overlay_png(ov, s, title, (start_time + t_scene) / total_scenes_dur, n)
            clip = TMP / f"c_{n:02d}_{k:03d}.mp4"
            first_of_scene, last_of_scene = (k == 0), (i == len(sents) - 1 and si == len(segs) - 1)
            if src_kind[0] == "slide":
                j = src_kind[1][i]; sl_len = slide_totals[j]; p0 = slide_pos[j]
                first = p0 == 0.0; slide_pos[j] += cd; last = abs(slide_pos[j] - sl_len) < 1e-6
                still_clip(FRAMES / f"slide_{slides[j]:02d}.png", ov, p0, p0 + cd, sl_len, clip, first, last, j + n)
            else:
                ui_clip(src_kind[1], ov, t_in_seg, cd, clip, i == 0, i == len(sents) - 1)
            parts.append(clip); t_scene += cd; t_in_seg += cd; k += 1
    lst = TMP / f"list_{n:02d}.txt"; lst.write_text("".join(f"file '{p}'\n" for p in parts))
    vid = TMP / f"scene_{n:02d}_v.mp4"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(vid)])
    final = CLIPS / f"scene_{n:02d}.mp4"
    run(["ffmpeg", "-y", "-i", str(vid), "-i", str(audio), "-af", "apad,aresample=44100", "-t", f"{ad:.3f}", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(final)])
    print(f"scene {n:02d}: {ad:6.1f}s  {len(parts)} caption cuts -> {final.name}")
    return ad


def make_music(seconds, out):
    """Soft ambient pad (A minor add9) generated with ffmpeg, no external assets."""
    notes = [110.0, 164.81, 220.0, 261.63, 329.63, 246.94]
    ins = []; fc = []
    for i, f in enumerate(notes):
        ins += ["-f", "lavfi", "-t", f"{seconds:.1f}", "-i", f"sine=frequency={f}:sample_rate=44100"]
        fc.append(f"[{i}:a]tremolo=f={0.1 + 0.05 * i:.2f}:d=0.6,volume=0.5[n{i}]")
    mix = "".join(f"[n{i}]" for i in range(len(notes))) + f"amix=inputs={len(notes)}:normalize=0,lowpass=f=900,aecho=0.8:0.7:900|1500:0.35|0.25,volume=0.5,afade=t=in:d=4,afade=t=out:st={seconds - 6:.1f}:d=6[m]"
    run(["ffmpeg", "-y", *ins, "-filter_complex", ";".join(fc) + ";" + mix, "-map", "[m]", "-ac", "2", str(out)])


def final_mix(video, out):
    d = dur(video); music = TMP / "music.wav"; make_music(d, music)
    af = ("[1:a]volume=0.55[bg];[0:a]asplit=2[v1][v2];[bg][v2]sidechaincompress=threshold=0.02:ratio=10:attack=40:release=700[ducked];"
          "[v1][ducked]amix=inputs=2:normalize=0:weights=1 1,loudnorm=I=-16:TP=-1.5:LRA=9[a]")
    run(["ffmpeg", "-y", "-i", str(video), "-i", str(music), "-filter_complex", af, "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--draft", action="store_true"); ap.add_argument("--scenes", type=int, nargs="*"); a = ap.parse_args()
    script = parse_script(); todo = a.scenes or sorted(SCENES)
    audios = {}
    for n in todo:
        real = AUDIO / f"scene_{n:02d}.wav"
        audios[n] = draft_audio(n, script[n]) if a.draft else (real if real.exists() else sys.exit(f"missing {real}"))
    total = sum(dur(audios[n]) + 0.6 for n in todo); t = 0.0
    for n in todo: t += build_scene(n, script[n], audios[n], total, t)
    lst = TMP / "all.txt"; lst.write_text("".join(f"file '{CLIPS / f'scene_{n:02d}.mp4'}'\n" for n in todo))
    name = "TeamTwins_demo_DRAFT.mp4" if a.draft else "TeamTwins_demo.mp4"
    joined = TMP / "joined.mp4"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(joined)])
    final_mix(joined, OUT / name)
    print(f"total {total / 60:.1f} min -> {OUT / name}")


if __name__ == "__main__":
    main()
