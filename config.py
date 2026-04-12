import os
from dotenv import load_dotenv

load_dotenv()

# ── Anthropic ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL       = "claude-3-haiku-20240307"

# ── Audio hardware ─────────────────────────────────────────────────────────────
MIC_DEVICE_INDEX   = int(os.getenv("MIC_DEVICE_INDEX", 1))
SAMPLE_RATE        = 44100
CHUNK_SIZE         = 1024

# ── Recording behaviour ────────────────────────────────────────────────────────
SILENCE_THRESHOLD_ENERGY   = 2000       # raw PCM energy level
SILENCE_THRESHOLD_DURATION = 3.5        # seconds of silence before stopping
NAME_SILENCE_DURATION      = 3.0        # shorter window for name capture
MIN_SPEECH_DURATION        = 1.0        # seconds; ignore clips shorter than this
MAX_RECORDING_DURATION     = 480        # 8 minutes hard cap per answer
NAME_MAX_DURATION          = 180        # 3 minutes for name/intro capture
NOISE_ADJUST_DURATION      = 1.0        # seconds to calibrate ambient noise

# ── TTS ────────────────────────────────────────────────────────────────────────
TTS_RATE = 150                          # words per minute for pyttsx3

# ── Whisper ────────────────────────────────────────────────────────────────────
WHISPER_MODEL      = "openai/whisper-large-v3"
WHISPER_LANGUAGE   = "en"
WHISPER_TARGET_SR  = 16000             # Whisper expects 16 kHz

# ── Interview structure ────────────────────────────────────────────────────────
PROFESSIONAL_QUESTION_COUNT = 6        # Phase 1 questions
HOBBY_QUESTION_COUNT        = 3        # Phase 3 questions
QUESTIONS_PER_CATEGORY      = 75       # generated per category
FOLLOWUP_PROBABILITY        = 0.6      # chance of follow-up on long answer
FOLLOWUP_MIN_WORDS          = 18       # answer must be at least this long
MAX_RETRIES_LISTEN          = 2        # recording retry attempts

# ── Willingness thresholds ─────────────────────────────────────────────────────
WILLINGNESS_LOW_THRESHOLD  = 30
WILLINGNESS_HIGH_THRESHOLD = 70

# ── Paths ──────────────────────────────────────────────────────────────────────
QUESTIONS_DIR  = "personality_questions"
RECORDINGS_DIR = "audio_recordings"
RESPONSES_DIR  = "personality_responses"