import time
from pathlib import Path

import pyttsx3

from config import TTS_RATE


def speak(text: str, rate: int = TTS_RATE) -> None:
    """Speak text aloud. Blocks until speech is complete."""
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.say(text)
    engine.runAndWait()


def speak_and_record(
    text: str,
    recording_dir: Path,
    recording_id: str,
    rate: int = TTS_RATE,
) -> tuple[Path | None, float]:
    """
    Speak text aloud and simultaneously save it to a WAV file.

    Returns
    -------
    (audio_file_path, first_byte_timestamp)
        audio_file_path   : Path to the saved WAV, or None on failure
        first_byte_timestamp : time.time() value when speech began
    """
    tts_start      = time.time()
    first_byte_time: float | None = None
    first_byte_flag = [False]          # mutable container for nonlocal write in callback

    def on_start(name: str) -> None:
        if not first_byte_flag[0]:
            first_byte_flag[0] = True
            nonlocal first_byte_time          # type: ignore[misc]
            first_byte_time = time.time()
            print(f"  [TTS] first byte after {first_byte_time - tts_start:.3f}s")

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        engine.connect("started-utterance", on_start)

        wav_path = recording_dir / f"{recording_id}_question.wav"
        engine.save_to_file(text, str(wav_path))
        engine.say(text)
        engine.runAndWait()

        if first_byte_time is None:
            first_byte_time = time.time()

        if wav_path.exists():
            return wav_path, first_byte_time
        else:
            print(f"  [TTS] Warning: WAV not created at {wav_path}")
            return None, first_byte_time

    except Exception as exc:
        print(f"  [TTS] Error in speak_and_record: {exc}")
        return None, time.time()


def record_system_speech(
    text: str,
    recording_dir: Path,
    recording_id: str,
    rate: int = TTS_RATE,
) -> Path | None:
    """
    Save spoken text to a WAV file without playing it aloud.

    Returns the path to the saved file, or None on failure.
    """
    try:
        wav_path = recording_dir / f"{recording_id}_question.wav"
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        engine.save_to_file(text, str(wav_path))
        engine.runAndWait()

        if wav_path.exists():
            return wav_path
        print(f"  [TTS] Warning: WAV not created at {wav_path}")
        return None

    except Exception as exc:
        print(f"  [TTS] Error in record_system_speech: {exc}")
        return None