from RealtimeSTT import AudioToTextRecorder

class STTService:
    def __init__(self, model_size="base.en"):
        self.recorder = AudioToTextRecorder(
            model=model_size,
            language="en",
            spinner=False
        )

    def demo_record(self, seconds=5):
        def print_text(text):
            print(f"[STT] {text}")

        print("Recording... Speak now!")
        self.recorder.start()
        self.recorder.text(print_text)
        self.recorder.stop()
        self.recorder.shutdown()
