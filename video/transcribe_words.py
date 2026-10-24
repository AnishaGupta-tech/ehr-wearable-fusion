import json, subprocess, sys
import numpy as np
from faster_whisper import WhisperModel
raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", sys.argv[1], "-ac", "1", "-ar", "16000", "-f", "f32le", "-"], capture_output=True, check=True).stdout
audio = np.frombuffer(raw, dtype=np.float32)
m = WhisperModel("small.en", device="cpu", compute_type="int8")
segs, _ = m.transcribe(audio, language="en", beam_size=1, word_timestamps=True)
words = [(round(w.start, 2), round(w.end, 2), w.word.strip()) for s in segs for w in s.words]
json.dump(words, open(sys.argv[2], "w")); print(len(words), "words")
