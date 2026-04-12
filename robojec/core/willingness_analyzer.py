from enum import Enum
from typing import Dict, Tuple

import librosa
import numpy as np

from config import (
    SAMPLE_RATE,
    WILLINGNESS_HIGH_THRESHOLD,
    WILLINGNESS_LOW_THRESHOLD,
)


class WillingnessLevel(Enum):
    LOW    = "low_willingness"
    MEDIUM = "medium_willingness"
    HIGH   = "high_willingness"


class WillingnessAnalyzer:
    """
    Estimates how engaged / willing a speaker is from raw audio data.

    Three signals are combined:
      - volume   : mean absolute amplitude
      - speech   : zero-crossing rate (proxy for voiced activity)
      - engagement: silence ratio + speaking duration bonus
    """

    _SPEECH_THRESHOLD = 0.01       # amplitude below this counts as silence
    _VOLUME_WEIGHT    = 0.30
    _SPEECH_WEIGHT    = 0.30
    _ENGAGE_WEIGHT    = 0.40

    def __init__(self) -> None:
        self.sample_rate = SAMPLE_RATE

    # ── public entry point ─────────────────────────────────────────────────────

    def analyze_audio_data(
        self, audio_data: np.ndarray
    ) -> Tuple[WillingnessLevel, float, Dict[str, float]]:
        """
        Analyse audio and return (level, composite_score, feature_scores).

        feature_scores keys: volume_score, speech_score, engagement_score
        """
        empty_scores = {"volume_score": 0.0, "speech_score": 0.0, "engagement_score": 0.0}

        if audio_data is None or (hasattr(audio_data, "size") and audio_data.size == 0) or len(audio_data) == 0:
            return WillingnessLevel.LOW, 0.0, empty_scores

        try:
            sr       = max(self.sample_rate, 1)
            duration = max(len(audio_data) / sr, 0.1)
            features = self._extract_features(audio_data)

            if features is None:
                return WillingnessLevel.LOW, 0.0, empty_scores

            scores    = self._feature_scores(features, duration)
            composite = self._composite(scores)
            level     = self._level(composite)

            print(
                f"  [Willingness] {level.value.upper()} "
                f"(score={composite:.1f} | "
                f"vol={scores['volume_score']:.1f} "
                f"speech={scores['speech_score']:.1f} "
                f"engage={scores['engagement_score']:.1f})"
            )
            return level, composite, scores

        except Exception as exc:
            print(f"  [Willingness] Analysis error: {exc}")
            neutral = {"volume_score": 50.0, "speech_score": 50.0, "engagement_score": 50.0}
            return WillingnessLevel.MEDIUM, 50.0, neutral

    # ── feature extraction ─────────────────────────────────────────────────────

    def _extract_features(self, audio_data: np.ndarray) -> Dict[str, float] | None:
        try:
            arr = np.abs(np.asarray(audio_data, dtype=float))

            vol_mean = float(np.mean(arr)) if arr.size else 0.0
            vol_max  = float(np.max(arr))  if arr.size else 0.0

            try:
                zcr            = librosa.feature.zero_crossing_rate(y=np.asarray(audio_data, dtype=float))
                speech_activity = float(np.mean(zcr)) if zcr.size else 0.0
            except Exception:
                speech_activity = 0.0

            is_silence      = arr < self._SPEECH_THRESHOLD
            silence_ratio   = float(np.mean(is_silence)) if is_silence.size else 1.0
            speech_segments = np.where(~is_silence)[0]
            sr              = max(self.sample_rate, 1)
            speech_duration = len(speech_segments) / sr

            return {
                "volume_mean":    vol_mean,
                "volume_max":     vol_max,
                "speech_activity": speech_activity,
                "silence_ratio":  silence_ratio,
                "speech_duration": speech_duration,
            }

        except Exception as exc:
            print(f"  [Willingness] Feature extraction error: {exc}")
            return None

    # ── scoring ────────────────────────────────────────────────────────────────

    def _feature_scores(
        self, features: Dict[str, float], speaking_duration: float
    ) -> Dict[str, float]:
        duration = max(speaking_duration, 0.1)
        return {
            "volume_score":     min(100.0, max(0.0, features.get("volume_mean", 0) * 1000)),
            "speech_score":     min(100.0, max(0.0, features.get("speech_activity", 0) * 500)),
            "engagement_score": max(
                0.0,
                100.0
                - features.get("silence_ratio", 1) * 100
                + min(100.0, duration * 10) * 0.2,
            ),
        }

    def _composite(self, scores: Dict[str, float]) -> float:
        return (
            scores.get("volume_score", 0)     * self._VOLUME_WEIGHT
            + scores.get("speech_score", 0)   * self._SPEECH_WEIGHT
            + scores.get("engagement_score", 0) * self._ENGAGE_WEIGHT
        )

    def _level(self, score: float) -> WillingnessLevel:
        if score < WILLINGNESS_LOW_THRESHOLD:
            return WillingnessLevel.LOW
        if score > WILLINGNESS_HIGH_THRESHOLD:
            return WillingnessLevel.HIGH
        return WillingnessLevel.MEDIUM