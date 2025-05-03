from RealtimeSTT import AudioToTextRecorder

class RTSTTService:
    """
    Asynchronous real-time speech-to-text service:
      push(samples) → internal thread calls RealtimeSTT
      samples should be 16kHz mono PCM int16 numpy array
    """
    def __init__(self, text_callback, model_size="base.en"):
        self.text_cb = text_callback
        self.recorder = AudioToTextRecorder(
            model=model_size,
            language="en",
            use_microphone=False,
            spinner=False,
            enable_realtime_transcription=True,
            on_realtime_transcription_update=self.text_cb
        )

    def push(self, samples):
        self.recorder.feed_audio(samples)

    def shutdown(self):
        self.recorder.shutdown()
