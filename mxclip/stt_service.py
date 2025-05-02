import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

class STTService:
    SAMPLE_RATE = 16000
    def __init__(self, model_size="base.en"):
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def _transcribe(self, audio):
        segments, _ = self.model.transcribe(
            audio, language="en", beam_size=1, vad_filter=True
        )
        for s in segments:
            print(f"[{s.start:.1f}s] {s.text}")

    def demo_record(self, seconds=5):
        frames = int(seconds * self.SAMPLE_RATE)
        rec = sd.rec(frames, samplerate=self.SAMPLE_RATE, channels=1, dtype="int16")
        sd.wait()
        samples = rec.flatten().astype(np.float32) / 32768.0
        self._transcribe(samples)
