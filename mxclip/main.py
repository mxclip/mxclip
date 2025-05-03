"""
mxclip CLI

Usage examples:
# Record microphone for N seconds and transcribe
python -m mxclip.main record --duration 5

# Play a local video file and transcribe its audio in real time
python -m mxclip.main video --file sample.mp4
"""
import argparse
from .stt_service import STTService


def _add_subcommands(parser: argparse.ArgumentParser) -> None:
    """Configure `record` and `video` sub‑commands."""
    sub = parser.add_subparsers(dest="cmd", required=True)

    # microphone demo
    rec = sub.add_parser("record", help="Record microphone and transcribe")
    rec.add_argument("--duration", type=int, default=5, help="Recording length in seconds")

    # video demo
    vid = sub.add_parser("video", help="Play local video and transcribe audio")
    vid.add_argument("--file", required=True, help="Path to local mp4 / mkv / flv file")


def _run_record(duration: int) -> None:
    stt = STTService()
    print(f"[mxclip] Recording for {duration} seconds… Speak now!")
    stt.demo_record(duration)


def _run_video(filepath: str) -> None:
    from .realtime_stt_service import RTSTTService
    from .shared_stream_listener import SharedStreamListener

    def on_text(text: str) -> None:
        print("[STT]", text, flush=True)

    stt = RTSTTService(on_text)
    listener = SharedStreamListener(filepath, push_audio=stt.push)
    listener.start()


def cli() -> None:
    parser = argparse.ArgumentParser(prog="mxclip")
    _add_subcommands(parser)
    args = parser.parse_args()

    if args.cmd == "record":
        _run_record(args.duration)
    elif args.cmd == "video":
        _run_video(args.file)


if __name__ == "__main__":
    cli()
