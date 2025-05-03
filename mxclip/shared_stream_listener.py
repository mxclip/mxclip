import ffmpeg
import numpy as np

class SharedStreamListener:
    """
    Read local video file -> output 0.5s audio chunks to push_audio callback
    """
    def __init__(self, filepath, push_audio, sample_rate=16000, chunk_sec=0.5):
        self.file = filepath
        self.push_audio = push_audio
        self.sample_rate = sample_rate
        self.chunk_sec = chunk_sec

    def start(self):
        try:
            process = (
                ffmpeg
                .input(self.file)
                .output('pipe:', format='s16le', ac=1, ar=str(self.sample_rate))
                .run_async(pipe_stdout=True, quiet=True)
            )
            bytes_per_chunk = int(self.sample_rate * self.chunk_sec * 2)  # int16 -> 2 bytes
            while True:
                data = process.stdout.read(bytes_per_chunk)
                if not data:
                    break
                audio_chunk = np.frombuffer(data, np.int16)
                self.push_audio(audio_chunk)
        except ffmpeg.Error as e:
            print(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            raise
        except Exception as e:
            print(f"Error processing audio stream: {str(e)}")
            raise
        finally:
            if 'process' in locals():
                process.kill()
