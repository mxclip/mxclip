import argparse
from .stt_service import STTService

def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=int, default=5, help="seconds to record")
    args = p.parse_args()
    stt = STTService()
    print(f"Recording {args.duration}s…  Speak now!")
    stt.demo_record(args.duration)

if __name__ == "__main__":
    cli()
