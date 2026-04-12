import time
from pathlib import Path
from typing import Optional

import numpy as np
import speech_recognition as sr
from faster_whisper import WhisperModel

from config import (
    MAX_RECORDING_DURATION,
    MIC_DEVICE_INDEX,
    MIN_SPEECH_DURATION,
    NAME_MAX_DURATION,
    NAME_SILENCE_DURATION,
    NOISE_ADJUST_DURATION,
    RECORDINGS_DIR,
    SILENCE_THRESHOLD_DURATION,
    SILENCE_THRESHOLD_ENERGY,
    WHISPER_LANGUAGE,
    MAX_RETRIES_LISTEN,
)
from robojec.utils.tts import speak

# ── faster-whisper (loaded once) ───────────────────────────────────────────────
_WHISPER = WhisperModel("base", device="cpu", compute_type="int8")

# ── meta-request detection ─────────────────────────────────────────────────────
# Phrases that mean the person wants the question repeated
_REPEAT_SIGNALS = {
    "can you repeat", "please repeat", "repeat that", "say that again",
    "could you repeat", "repeat please", "say again", "pardon",
    "come again", "once more", "repeat the question", "say it again",
    "could you say that again", "please say that again",
}
# Phrases that mean the person wants a rephrasing / clarification
_REPHRASE_SIGNALS = {
    "what do you mean", "i don't understand", "i didn't understand",
    "could you explain", "can you explain", "what does that mean",
    "i'm not sure what you mean", "not sure i understand",
    "could you rephrase", "can you rephrase", "rephrase that",
    "i don't get it", "i didn't get that", "what exactly",
    "could you clarify", "can you clarify", "not clear",
    "couldn't understand", "couldn't quite understand",
    "didn't quite understand", "not quite clear",
    "i couldn't understand", "i didn't quite understand",
    "couldn't hear", "didn't hear it clearly",
    "sorry i couldn't", "sorry couldn't", "i couldn't quite",
    "unclear", "not sure i follow",
}
# Single keywords that on their own (in a short utterance) indicate rephrase
_REPHRASE_KEYWORDS = {
    "understand", "understanding", "unclear", "confused", "confusing",
    "huh", "pardon", "clarity", "clearer", "simpler",
}

_EXIT_COMMANDS = {
    "quit", "end", "stop", "exit",
    "end interview", "exit interview",
    "stop interview", "quit interview",
}


class MetaRequest:
    """Returned instead of a string when a meta-request is detected."""
    REPEAT   = "repeat"
    REPHRASE = "rephrase"

    def __init__(self, kind: str):
        self.kind = kind

    def __repr__(self):
        return f"MetaRequest({self.kind})"


def detect_meta_request(text: str) -> Optional[MetaRequest]:
    """
    Check if text is a repeat or rephrase request rather than an answer.
    Also catches short utterances like 'I didn't understand it clearly'
    that are below the normal word count threshold but clearly meta-requests.
    """
    if not text:
        return None
    t = text.lower().strip()

    # check repeat signals first
    if any(s in t for s in _REPEAT_SIGNALS):
        return MetaRequest(MetaRequest.REPEAT)

    # check rephrase phrase signals
    if any(s in t for s in _REPHRASE_SIGNALS):
        return MetaRequest(MetaRequest.REPHRASE)

    # short utterances (< 8 words) containing rephrase keywords
    words = t.split()
    if len(words) <= 8 and any(kw in words for kw in _REPHRASE_KEYWORDS):
        return MetaRequest(MetaRequest.REPHRASE)

    # catch "sorry" + any confusion word in same short utterance
    if len(words) <= 10 and "sorry" in words:
        if any(kw in t for kw in ["understand", "clear", "get it", "follow", "hear"]):
            return MetaRequest(MetaRequest.REPHRASE)

    return None


# ── directory helpers ──────────────────────────────────────────────────────────

def create_user_recording_directory(user_name: str) -> Path:
    timestamp     = time.strftime("%Y%m%d_%H%M%S")
    safe_name     = user_name.lower().replace(" ", "_")
    recording_dir = Path(RECORDINGS_DIR) / safe_name / timestamp
    recording_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [Audio] Recordings → {recording_dir}")
    return recording_dir


# ── internal helpers ───────────────────────────────────────────────────────────

def _pcm_energy(raw_data: bytes) -> float:
    if not raw_data:
        return 0.0
    n = len(raw_data) // 2
    return sum(
        int.from_bytes(raw_data[i : i + 2], byteorder="little", signed=True) ** 2
        for i in range(0, len(raw_data), 2)
    ) / max(n, 1)


def _record_frames(
    source, recognizer, silence_threshold, min_speech_duration, max_duration
):
    frames          = []
    frame_duration  = 0.1
    start_time      = time.time()
    last_sound_time = start_time

    while True:
        now             = time.time()
        elapsed         = now - start_time
        silence_elapsed = now - last_sound_time

        print(
            f"\r  [Rec] {elapsed:.1f}s | silence {silence_elapsed:.1f}s/{silence_threshold}s",
            end="", flush=True,
        )

        if elapsed >= max_duration:
            print("\n  [Rec] Max duration reached.")
            break
        if silence_elapsed >= silence_threshold and elapsed > min_speech_duration:
            print("\n  [Rec] Silence threshold reached.")
            break

        try:
            frame = recognizer.record(source, duration=frame_duration)
            frames.append(frame)
            if _pcm_energy(frame.get_raw_data()) > SILENCE_THRESHOLD_ENERGY:
                last_sound_time = now
        except sr.WaitTimeoutError:
            continue
        except Exception as exc:
            print(f"\n  [Rec] Frame error: {exc}")
            time.sleep(0.2)

    return frames


def _frames_to_audio(frames, source):
    if not frames:
        return None
    raw = b"".join(f.get_raw_data() for f in frames)
    return sr.AudioData(raw, source.SAMPLE_RATE, source.SAMPLE_WIDTH) if raw else None


def _save_wav(audio, path: Path) -> None:
    try:
        with open(path, "wb") as fh:
            fh.write(audio.get_wav_data())
    except Exception as exc:
        print(f"  [Audio] Save error: {exc}")


def _faster_whisper_transcribe(audio_np: np.ndarray, orig_sr: int) -> str:
    import librosa
    if orig_sr != 16000:
        try:
            audio_np = librosa.resample(audio_np, orig_sr=orig_sr, target_sr=16000)
        except Exception:
            pass
    segments, _ = _WHISPER.transcribe(
        audio_np,
        language=WHISPER_LANGUAGE,
        beam_size=1,
        vad_filter=True,
    )
    return " ".join(seg.text for seg in segments).strip()


# ── public recording functions ─────────────────────────────────────────────────

def listen_and_save(
    recording_dir: Path,
    question_id: str,
    silence_threshold: float = SILENCE_THRESHOLD_DURATION,
    min_speech_duration: float = MIN_SPEECH_DURATION,
    max_recording_duration: float = MAX_RECORDING_DURATION,
):
    """
    Record and transcribe with faster-whisper.

    Returns one of:
      (text, audio_np, processing_end_time)   — normal answer
      ("quit", audio_np, time)                — exit command
      (MetaRequest, audio_np, time)           — repeat/rephrase request
    """
    recognizer = sr.Recognizer()

    for attempt in range(MAX_RETRIES_LISTEN):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        print(f"  [Listen] Attempt {attempt + 1}/{MAX_RETRIES_LISTEN}")

        try:
            time.sleep(0.3)
            with sr.Microphone(device_index=MIC_DEVICE_INDEX) as source:
                recognizer.adjust_for_ambient_noise(source, duration=NOISE_ADJUST_DURATION)
                recognizer.dynamic_energy_threshold = False
                recognizer.energy_threshold         = SILENCE_THRESHOLD_ENERGY

                try:
                    frames = _record_frames(
                        source, recognizer,
                        silence_threshold, min_speech_duration, max_recording_duration,
                    )
                except KeyboardInterrupt:
                    return "quit", np.array([], dtype=np.float32), time.time()

                audio = _frames_to_audio(frames, source)
                if audio is None:
                    print("  [Listen] No audio captured.")
                    continue

                if recording_dir:
                    _save_wav(audio, recording_dir / f"{timestamp}_{question_id}_a{attempt+1}.wav")

                audio_np = (
                    np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32)
                    / 32768.0
                )

                try:
                    text           = _faster_whisper_transcribe(audio_np, source.SAMPLE_RATE)
                    processing_end = time.time()
                except Exception as exc:
                    print(f"  [Whisper] Error: {exc}")
                    if attempt < MAX_RETRIES_LISTEN - 1:
                        speak("I'm having trouble understanding. Could you please try again?")
                        continue
                    return "", np.array([], dtype=np.float32), time.time()

                print(f"  [You] {text}")

                if not text:
                    if attempt < MAX_RETRIES_LISTEN - 1:
                        speak("I didn't quite catch that. Could you please say that again?")
                        continue
                    return "", np.array([], dtype=np.float32), processing_end

                # exit command
                if text.lower().strip() in _EXIT_COMMANDS:
                    return "quit", audio_np, processing_end

                # meta-request: check BEFORE word count gate
                # short utterances like "I didn't understand" must be caught here
                meta = detect_meta_request(text)
                if meta:
                    return meta, audio_np, processing_end

                # only apply word count gate for genuine answers
                if len(text.split()) < 3:
                    if attempt < MAX_RETRIES_LISTEN - 1:
                        speak("I didn't quite catch that. Could you please say that again?")
                        continue
                    return "", np.array([], dtype=np.float32), processing_end

                return text, audio_np, processing_end

        except Exception as exc:
            print(f"  [Listen] Outer error: {exc}")
            if attempt < MAX_RETRIES_LISTEN - 1:
                speak("There was a problem. Could you try again?")
                time.sleep(1.0)
                continue

    return "", np.array([], dtype=np.float32), time.time()


def listen_and_save_name(
    recording_dir,
    question_id: str,
    silence_threshold: float = NAME_SILENCE_DURATION,
    min_speech_duration: float = 1.0,
    max_recording_duration: float = NAME_MAX_DURATION,
    use_whisper: bool = False,
):
    """
    Short-window recording for name/profession collection.

    use_whisper=False (default): Google SR — fast, good for wake window
    use_whisper=True           : faster-whisper — more accurate, used for
                                 name capture and profession collection where
                                 accuracy matters more than speed

    Returns (text, audio_np)
    """
    recognizer = sr.Recognizer()

    try:
        time.sleep(0.3)
        with sr.Microphone(device_index=MIC_DEVICE_INDEX) as source:
            recognizer.adjust_for_ambient_noise(source, duration=NOISE_ADJUST_DURATION)
            recognizer.dynamic_energy_threshold = False
            recognizer.energy_threshold         = SILENCE_THRESHOLD_ENERGY

            try:
                frames = _record_frames(
                    source, recognizer,
                    silence_threshold, min_speech_duration, max_recording_duration,
                )
            except KeyboardInterrupt:
                return "quit", np.array([], dtype=np.float32)

            audio = _frames_to_audio(frames, source)
            if audio is None:
                return "", np.array([], dtype=np.float32)

            audio_np = (
                np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32)
                / 32768.0
            )

            if recording_dir:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                _save_wav(audio, recording_dir / f"{timestamp}_{question_id}.wav")

            if use_whisper:
                # Use faster-whisper for better accuracy
                try:
                    text = _faster_whisper_transcribe(audio_np, source.SAMPLE_RATE)
                    print(f"  [You] {text}")
                except Exception as exc:
                    print(f"  [Whisper] Error in name capture: {exc}")
                    return "", audio_np
            else:
                # Use Google SR for speed (wake window, quick checks)
                try:
                    text = recognizer.recognize_google(audio)
                    print(f"  [You] {text}")
                except sr.UnknownValueError:
                    return "", audio_np
                except Exception as exc:
                    print(f"  [Google SR] Error: {exc}")
                    return "", audio_np

            if not text or not text.strip():
                return "", audio_np

            if text.lower().strip() in _EXIT_COMMANDS:
                return "quit", audio_np

            return text, audio_np

    except Exception as exc:
        print(f"  [listen_and_save_name] Error: {exc}")
        return "", np.array([], dtype=np.float32)