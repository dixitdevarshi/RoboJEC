if __name__ == "__main__":
    MY_API_KEY = "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"
   
    # Define a wrapper function to include your API key
    def start_voice_activated_system():
        try:
            run_with_voice_activation()
        except Exception as e:
            print(f"Error in voice activation system: {e}")
            # Fall back to standard interview if voice activation fails
            run_interview(api_key=MY_API_KEY)
   
    # Call the wrapper function
    start_voice_activated_system()

def listen_for_trigger(trigger_phrase="hey robo", timeout=None):
    """
    Continuously listen for the trigger phrase.
    Returns True if just the trigger phrase is heard, or the full text after the trigger phrase if found.
    """
    print(f"System in sleep mode. Listening for trigger phrase: '{trigger_phrase}'...")
    recognizer = sr.Recognizer()
   
    start_time = time.time()
   
    while timeout is None or (time.time() - start_time) < timeout:
        try:
            with sr.Microphone(device_index=2) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
               
                # Short timeout to allow checking for program termination
                audio = recognizer.listen(source, timeout=6, phrase_time_limit=10)
               
                try:
                    text = recognizer.recognize_google(audio).lower()
                    print(f"Heard: {text}")
                   
                    if trigger_phrase in text:
                        print(f"Trigger phrase detected!")
                        # Check if there's anything after the trigger phrase
                        after_trigger = text.split(trigger_phrase, 1)[1].strip()
                        if after_trigger:
                            # If there's content after the trigger, return it
                            return after_trigger
                        else:
                            # If it's just the trigger phrase, return a special signal
                            return True
                except sr.UnknownValueError:
                    # Speech wasn't recognized - continue listening
                    pass
                except sr.RequestError as e:
                    print(f"Could not request results: {e}")
                    time.sleep(2)  # Wait before retrying
       
        except Exception as e:
            print(f"Error in listening loop: {e}")
            time.sleep(1)  # Prevent tight error loops
   
    return ""


def modify_run_interview():
    """
    This is a wrapper function that maintains the original signature of run_interview()
    but ensures the system is properly activated first.
    """
    # Instead of directly calling run_interview, call this function
    run_with_voice_activation()


def run_with_voice_activation():
    """Main function that waits for trigger and then runs the interview."""
    print("Voice activation system started.")
    print("Say 'Hey RoboJec' followed by an introduction to start the interview.")
    print("For example: 'Hey RoboJec, this is John' or 'Hey RoboJec, I'd like to start an interview'")
   
    # Get API key from the global scope
    global MY_API_KEY
   
    try:
        while True:
            # Wait for trigger phrase and capture introduction
            intro_text = listen_for_trigger()
           
            if intro_text:
                # Activation confirmation
                activation_responses = ["Hello, good to see you!",
                                        "Hi there, wonderful to meet you!",
                                        "Greetings, it's a pleasure!",
                                        "Hello, what a delight to meet you!"
                                        ]
                activation_msg = random.choice(activation_responses)
                print(activation_msg)
                speak(activation_msg)
               
                # Run the interview with your API key
                run_interview(api_key=MY_API_KEY)
               
                # After interview completes, go back to listening mode
                print("\nInterview completed. Returning to sleep mode...")
                sleep_msg = "Thank you for using the interview system."
                print(sleep_msg)
                speak(sleep_msg)
               
                # Optional: add a delay before starting to listen again
                time.sleep(2)
           
    except KeyboardInterrupt:
        print("\nVoice activation system terminated.")


def speak(text, rate=150):  # Keep this original function
    """Convert text to speech at a slower speed."""
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)  
   
    engine.say(text)
    engine.runAndWait()

def record_system_speech(text, recording_dir, recording_id, rate=150):
    """
    Record the system's speech to a WAV file.
    
    Args:
        text (str): The text to be spoken and recorded
        recording_dir (Path): Directory to save the recording
        recording_id (str): Identifier for the recording
        rate (int): Speech rate
        
    Returns:
        Path: Path to the saved audio file or None if recording failed
    """
    try:
        # Create a temporary WAV file
        temp_file = recording_dir / f"{recording_id}_question.wav"
        
        # Initialize the TTS engine
        engine = pyttsx3.init()
        engine.setProperty('rate', rate)
        
        # Save speech to file instead of playing it
        print(f"Recording system question to {temp_file}...")
        engine.save_to_file(text, str(temp_file))
        engine.runAndWait()
        
        # Verify file was created
        if temp_file.exists():
            print(f"System question audio saved to {temp_file}")
            return temp_file
        else:
            print("Failed to save system question audio")
            return None
            
    except Exception as e:
        print(f"Error recording system speech: {str(e)}")
        return None

def speak_and_record(text, recording_dir, recording_id, rate=150):
    """
    Speak the text aloud and record it at the same time.
    Return the timing of first byte production.
    
    Args:
        text (str): The text to be spoken and recorded
        recording_dir (Path): Directory to save the recording
        recording_id (str): Identifier for the recording
        rate (int): Speech rate
        
    Returns:
        tuple: (Path to audio file, timestamp of first byte production)
    """
    # Mark when we start the TTS process
    tts_start_time = time.time()
    
    # Create a flag to track when first byte is produced
    first_byte_produced = False
    first_byte_time = None
    
    # Define callback for when engine starts producing speech
    def onStart(name):
        nonlocal first_byte_produced, first_byte_time
        if not first_byte_produced:
            first_byte_time = time.time()
            first_byte_produced = True
            print(f"First speech byte produced after {first_byte_time - tts_start_time:.3f}s")
    
    try:
        # Initialize the TTS engine
        engine = pyttsx3.init()
        engine.setProperty('rate', rate)
        
        # Register the callback
        engine.connect('started-utterance', onStart)
        
        # Create a temporary WAV file
        temp_file = recording_dir / f"{recording_id}_question.wav"
        
        # Save speech to file
        print(f"Recording system question to {temp_file}...")
        engine.save_to_file(text, str(temp_file))
        
        # Speak the text
        engine.say(text)
        engine.runAndWait()
        
        # If callback didn't fire, use end time as fallback
        if not first_byte_produced:
            first_byte_time = time.time()
            print("Warning: Could not detect first byte production, using completion time")
        
        # Verify file was created
        if temp_file.exists():
            print(f"System question audio saved to {temp_file}")
            return temp_file, first_byte_time
        else:
            print("Failed to save system question audio")
            return None, first_byte_time
            
    except Exception as e:
        print(f"Error in speak_and_record: {str(e)}")
        return None, time.time()

def create_user_recording_directory(user_name):
    """Create a directory for storing user's audio recordings."""
    # Create a unique folder name using timestamp to avoid overwriting previous sessions
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    recording_dir = Path(f"./audio_recordings/{user_name.lower().replace(' ', '_')}/{timestamp}")
    recording_dir.mkdir(parents=True, exist_ok=True)
    print(f"Audio recordings will be saved to: {recording_dir}")
    return recording_dir
