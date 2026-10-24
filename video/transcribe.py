import json, subprocess, sys
import numpy as np
from faster_whisper import WhisperModel
raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", sys.argv[1], "-ac", "1", "-ar", "16000", "-f", "f32le", "-"], capture_output=True, check=True).stdout
audio = np.frombuffer(raw, dtype=np.float32)
m = WhisperModel("small.en", device="cpu", compute_type="int8")
segs, info = m.transcribe(audio, language="en", beam_size=1)
out = [(round(s.start, 2), round(s.end, 2), s.text.strip()) for s in segs]
json.dump(out, open(sys.argv[2], "w"))
print(len(out), "segments, audio", round(len(audio) / 16000, 1), "s")
