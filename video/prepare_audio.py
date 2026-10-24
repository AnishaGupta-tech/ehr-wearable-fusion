"""Clean the raw recording: denoise, trim start noise, remove pauses, even out speaking speed, cut into scenes.
Writes video/audio/scene_XX.wav and video/tmp/scene_XX_words.json (word times on the final timeline)."""
import json, re, statistics, subprocess, sys, wave
import numpy as np

HERE = __import__("pathlib").Path(__file__).resolve().parent
RAW = HERE / "audio" / "Team twins.m4a"
SPEED = float(sys.argv[1]) if len(sys.argv) > 1 else 1.1     # overall pace-up
GAP = 0.14                                                    # silence kept between phrases (s)
SR = 44100
W = [tuple(w) for w in json.load(open(HERE / "tmp" / "words.json"))]


def run_bytes(cmd, data=None):
    r = subprocess.run(cmd, input=data, capture_output=True)
    if r.returncode: sys.exit(r.stderr.decode()[-800:])
    return r.stdout


# 1. denoise the whole file (timing unchanged)
clean = run_bytes(["ffmpeg", "-loglevel", "error", "-i", str(RAW), "-ac", "1", "-ar", str(SR),
                   "-af", "highpass=f=75,afftdn=nr=14:nf=-32:tn=1,acompressor=threshold=-24dB:ratio=2.5:attack=15:release=250:makeup=3",
                   "-f", "f32le", "-"])
x = np.frombuffer(clean, dtype=np.float32)

# 2. phrases from word gaps (drops the noisy start and all pauses)
ph = [[0]]
for i in range(1, len(W)):
    if W[i][0] - W[i - 1][1] >= 0.28: ph.append([i])
    else: ph[-1].append(i)
merged = []
for p in ph:
    if merged and (W[p[0]][0] - W[merged[-1][-1]][1] < 0.7) and (W[merged[-1][-1]][1] - W[merged[-1][0]][0] < 2.2):
        merged[-1] += p
    else: merged.append(p)
ph = merged


def rate(p): return len(p) / max(W[p[-1]][1] - W[p[0]][0], 0.3)


target = SPEED * statistics.median(rate(p) for p in ph if len(p) >= 4)
pieces, newW, t = [], [], 0.0
for k, p in enumerate(ph):
    ps, pe = W[p[0]][0], W[p[-1]][1]
    prev_end = W[ph[k - 1][-1]][1] if k else ps - 0.3
    next_start = W[ph[k + 1][0]][0] if k + 1 < len(ph) else pe + 0.3
    a = ps - min(0.10, (ps - prev_end) / 2 - 0.01); b = pe + min(0.12, (next_start - pe) / 2 - 0.01)
    a = max(a, 0); seg = x[int(a * SR):int(b * SR)]
    tempo = float(np.clip(target / rate(p), 1.0 if SPEED >= 1 else 0.9, 1.35))
    if abs(tempo - 1) > 0.01:
        seg = np.frombuffer(run_bytes(["ffmpeg", "-loglevel", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", "-af", f"atempo={tempo:.4f}", "-f", "f32le", "-"], seg.tobytes()), dtype=np.float32)
    for i in p: newW.append((t + (W[i][0] - a) / tempo, t + (W[i][1] - a) / tempo, W[i][2]))
    pieces.append(seg); pieces.append(np.zeros(int(GAP * SR), dtype=np.float32)); t += len(seg) / SR + GAP
tight = np.concatenate(pieces)
pcm = run_bytes(["ffmpeg", "-loglevel", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-", "-af", "loudnorm=I=-17:TP=-2:LRA=7", "-ar", str(SR), "-f", "s16le", "-"], tight.tobytes())
tight_s = np.frombuffer(pcm, dtype=np.int16)
print(f"raw {len(x) / SR / 60:.1f} min -> tight {len(tight_s) / SR / 60:.1f} min (speed target x{SPEED}, {len(ph)} phrases)")

# 3. scene boundaries
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
toks = [norm(w[2]) for w in newW]
marks = {1: "hello and welcome we are", 2: "let us start with why this matters", 3: "before the technology we fixed", 4: "this challenge is organised by", 5: "now the data we use",
         6: "this is how it all fits", 7: "the event model uses", 8: "now the results on", 9: "now the part that matters", 10: "now a word about trust",
         11: "now what does this mean", 12: "the whole system is about"}
def find(phrase, after):
    p = [norm(q) for q in phrase.split()]
    for i in range(after, len(toks) - len(p)):
        if sum(toks[i + k] == p[k] for k in range(len(p))) >= len(p) - 1: return i
    for i in range(after, len(toks) - 3):
        if toks[i:i + 3] == p[:3]: return i
    sys.exit("mark not found: " + phrase)
idx, pos = {}, 0
for n, ph_ in marks.items():
    idx[n] = find(ph_, pos); pos = idx[n]
cut = {1: 0.0}
for n in range(2, 13): cut[n] = (newW[idx[n] - 1][1] + newW[idx[n]][0]) / 2
total = len(tight_s) / SR
for n in range(1, 13):
    a, b = cut[n], (cut[n + 1] if n < 12 else total)
    seg = tight_s[int(a * SR):int(b * SR)]
    with wave.open(str(HERE / "audio" / f"scene_{n:02d}.wav"), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(SR); f.writeframes(seg.tobytes())
    lo, hi = idx[n], (idx[n + 1] if n < 12 else len(newW))
    json.dump([(round(s - a, 3), round(e - a, 3), w) for s, e, w in newW[lo:hi]], open(HERE / "tmp" / f"scene_{n:02d}_words.json", "w"))
    print(f"scene {n:02d}: {(b - a):6.1f} s")
