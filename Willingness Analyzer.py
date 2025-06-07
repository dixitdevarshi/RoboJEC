class WillingnessAnalyzer:
    def __init__(self):
        # Audio configuration
        self.sample_rate = 44100  # Standard sample rate for audio processing
        self.chunk_size = 1024  # Number of frames per buffer
        self.speech_threshold = 0.01  # Minimum amplitude to consider as speech
        self.silence_threshold_duration = 1.5  # Seconds of silence to consider as pause
        self.min_speech_duration = 0.3  # Minimum duration to consider as valid speech
       
        # Willingness score thresholds
        self.low_threshold = 30  # Scores below this are LOW willingness
        self.high_threshold = 70  # Scores above this are HIGH willingness
       
        # Feature weights for composite score calculation
        self.volume_weight = 0.3
        self.speech_activity_weight = 0.3
        self.engagement_weight = 0.4
   
    def analyze_audio_data(self, audio_data):
        """Analyze audio data to determine user's willingness level."""
        # Check for empty or invalid audio data
        if audio_data is None or len(audio_data) == 0 or (hasattr(audio_data, 'size') and audio_data.size == 0):
            return WillingnessLevel.LOW, 0, {
                'volume_score': 0,
                'speech_score': 0,
                'engagement_score': 0
            }
       
        try:
            # Ensure sample_rate is valid
            if self.sample_rate <= 0:
                self.sample_rate = 44100  # Use default if zero or negative
               
            # Calculate speaking duration with protection
            speaking_duration = max(0.1, len(audio_data) / max(1, self.sample_rate))
           
            # Extract features
            features = self._extract_audio_features(audio_data)
            if features is None:
                return WillingnessLevel.LOW, 0, {
                    'volume_score': 0,
                    'speech_score': 0,
                    'engagement_score': 0
                }
               
            # Calculate individual feature scores
            feature_scores = self._calculate_feature_scores(features, speaking_duration)
           
            # Calculate composite willingness score (0-100)
            composite_score = self._calculate_composite_score(feature_scores)
           
            # Determine willingness level as enum
            if composite_score < self.low_threshold:
                willingness_level = WillingnessLevel.LOW
            elif composite_score > self.high_threshold:
                willingness_level = WillingnessLevel.HIGH
            else:
                willingness_level = WillingnessLevel.MEDIUM
               
            return willingness_level, composite_score, feature_scores
        except Exception as e:
            print(f"Error in willingness analysis: {e}")
            return WillingnessLevel.MEDIUM, 50, {
                'volume_score': 50,
                'speech_score': 50,
                'engagement_score': 50
            }
   
    def _extract_audio_features(self, audio_data):
        """Extract relevant features from audio data."""
        features = {}
       
        # Check for empty audio
        if audio_data is None or len(audio_data) == 0 or (hasattr(audio_data, 'size') and audio_data.size == 0):
            return {
                'volume_mean': 0.0,
                'volume_max': 0.0,
                'speech_activity': 0.0,
                'silence_ratio': 1.0,
                'speech_duration': 0.0
            }
       
        try:
            # Handle both numpy arrays and lists
            if hasattr(audio_data, 'size') and audio_data.size > 0:
                abs_audio = np.abs(audio_data)
            elif len(audio_data) > 0:
                # Convert to numpy array if it's a list or other sequence
                abs_audio = np.abs(np.array(audio_data, dtype=float))
            else:
                abs_audio = np.array([0.0])
               
            # Calculate volume features with protection
            features['volume_mean'] = float(np.mean(abs_audio)) if len(abs_audio) > 0 else 0.0
            features['volume_max'] = float(np.max(abs_audio)) if len(abs_audio) > 0 else 0.0
           
            # Calculate speech activity with protection
            try:
                if len(audio_data) > 0:
                    zcr = librosa.feature.zero_crossing_rate(y=np.array(audio_data, dtype=float))
                    features['speech_activity'] = float(np.mean(zcr)) if zcr.size > 0 else 0.0
                else:
                    features['speech_activity'] = 0.0
            except Exception as e:
                print(f"Error calculating speech activity: {e}")
                features['speech_activity'] = 0.0
               
            # Silence ratio calculation with protection
            is_silence = abs_audio < self.speech_threshold
            features['silence_ratio'] = float(np.mean(is_silence)) if len(is_silence) > 0 else 1.0
           
            # Speech duration calculation with protection
            speech_segments = np.where(~is_silence)[0] if len(is_silence) > 0 else np.array([])
           
            # Protect against zero sample_rate
            safe_sample_rate = max(1, self.sample_rate)  # Ensure sample_rate is never zero
           
            features['speech_duration'] = len(speech_segments) / safe_sample_rate
               
            return features
        except Exception as e:
            print(f"Error in audio feature extraction: {str(e)}")
            return None
   
    def _calculate_feature_scores(self, features, speaking_duration):
        """Calculate normalized scores (0-100) for each relevant feature."""
        scores = {}
       
        # Ensure speaking_duration has a minimum value
        speaking_duration = max(0.1, speaking_duration)
       
        # 1. Volume Score (0-100) - with safety checks
        volume_mean = features.get('volume_mean', 0)
        scores['volume_score'] = min(100, max(0, volume_mean * 1000))
       
        # 2. Speech Activity Score (0-100) - with safety checks
        speech_activity = features.get('speech_activity', 0)
        scores['speech_score'] = min(100, max(0, speech_activity * 500))
       
        # 3. Engagement Score (0-100) - with safety checks
        silence_ratio = features.get('silence_ratio', 1)
        silence_penalty = silence_ratio * 100
       
        # Ensure minimum value for speaking_duration
        duration_bonus = min(100, speaking_duration * 10)  # 10s = max bonus
        scores['engagement_score'] = max(0, 100 - silence_penalty + (duration_bonus * 0.2))
       
        return scores
   
    def _calculate_composite_score(self, feature_scores):
        """
        Calculate composite willingness score from individual feature scores.
        Args:
            feature_scores: Dictionary of normalized feature scores
        Returns:
            float: Composite score (0-100)
        """
        # Use get with default values to handle missing keys
        volume_score = feature_scores.get('volume_score', 0)
        speech_score = feature_scores.get('speech_score', 0)
        engagement_score = feature_scores.get('engagement_score', 0)
       
        return (
            volume_score * self.volume_weight +
            speech_score * self.speech_activity_weight +
            engagement_score * self.engagement_weight
        )
   
    def _determine_willingness_level(self, composite_score):
        """
        Determine willingness level based on composite score.
        Args:
            composite_score: Calculated willingness score (0-100)
        Returns:
            WillingnessLevel: Enum value representing willingness level
        """
        if composite_score < self.low_threshold:
            return WillingnessLevel.LOW
        elif composite_score > self.high_threshold:
            return WillingnessLevel.HIGH
        else:
            return WillingnessLevel.MEDIUM
   
    def analyze_features(self, audio_data):
        """
        Wrapper method for feature extraction (maintained for backward compatibility).
        """
        return self._extract_audio_features(audio_data)
   
    def determine_willingness(self, features, speaking_duration):
        """
        Wrapper method for willingness determination (maintained for backward compatibility).
        """
        if features is None or speaking_duration < self.min_speech_duration:
            return "low", 0, {
                'volume_score': 0,
                'speech_score': 0,
                'engagement_score': 0
            }
           
        feature_scores = self._calculate_feature_scores(features, speaking_duration)
        composite_score = self._calculate_composite_score(feature_scores)
        level = self._determine_willingness_level(composite_score).value
       
        return level, composite_score, feature_scores
