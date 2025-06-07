WHISPER_PIPELINE = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)


def sr_audio_to_whisper_format(audio_data):
    """Convert SpeechRecognition AudioData to Whisper-compatible format"""
    audio_np = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
    return audio_np, audio_data.sample_rate

def listen_and_save(recording_dir, question_id, silence_threshold=3, min_speech_duration=1, max_recording_duration=480):
    max_retries = 2
    current_retry = 0
    recognizer = sr.Recognizer()
   
    while current_retry < max_retries:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        print(f"Starting recording attempt {current_retry+1}/{max_retries}...")
       
        try:
            time.sleep(0.5)
            with sr.Microphone(device_index=2) as source:
                print("Adjusting for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                print("Recording started. Speak now...")
               
                recognizer.dynamic_energy_threshold = False
                recognizer.energy_threshold = 2000
                recognizer.pause_threshold = silence_threshold
               
                audio_frames = []
                frame_duration = 0.1
                start_time = time.time()
                last_sound_time = start_time
               
                try:
                    while True:
                        current_time = time.time()
                        elapsed = current_time - start_time
                        silence_duration = current_time - last_sound_time
                       
                        print(f"\rRecording: {elapsed:.1f}s | Silence: {silence_duration:.1f}s/{silence_threshold}s", end="", flush=True)
                       
                        if elapsed >= max_recording_duration:
                            print("\nMaximum recording duration reached!")
                            break
                       
                        if silence_duration >= silence_threshold and elapsed > min_speech_duration:
                            print("\nSilence threshold reached!")
                            break
                       
                        try:
                            frame = recognizer.record(source, duration=frame_duration)
                            audio_frames.append(frame)
                           
                            frame_data = frame.get_raw_data()
                            if len(frame_data) == 0:
                                continue
                           
                            energy = sum(
                                int.from_bytes(frame_data[i:i+2], byteorder='little', signed=True)**2
                                for i in range(0, len(frame_data), 2)
                            ) / (len(frame_data)/2)
                           
                            if energy > 2000:
                                last_sound_time = current_time
                       
                        except sr.WaitTimeoutError:
                            continue
                        except Exception as e:
                            print(f"\nError during frame recording: {str(e)}")
                            time.sleep(0.2)
                            continue
               
                except KeyboardInterrupt:
                    print("\nRecording interrupted by user")
                    return "quit", np.array([], dtype=np.float32), time.time()
               
                print("\nProcessing recorded audio...")
               
                if not audio_frames:
                    print("No audio frames recorded")
                    current_retry += 1
                    continue
               
                try:
                    combined_audio = sr.AudioData(
                        b''.join(frame.get_raw_data() for frame in audio_frames),
                        source.SAMPLE_RATE,
                        source.SAMPLE_WIDTH
                    )
                   
                    if len(combined_audio.get_raw_data()) == 0:
                        print("Warning: Empty audio data received")
                        current_retry += 1
                        continue
                   
                    if recording_dir:
                        filename = f"{timestamp}_{question_id}_attempt{current_retry+1}.wav"
                        filepath = recording_dir / filename
                        try:
                            with open(filepath, "wb") as f:
                                f.write(combined_audio.get_wav_data())
                            print(f"Audio saved to {filepath}")
                        except Exception as e:
                            print(f"Error saving audio file: {str(e)}")
                   
                    audio_np = np.frombuffer(combined_audio.get_raw_data(), dtype=np.int16).astype(np.float32) / 32768.0
                   
                    try:
                        audio_resampled = librosa.resample(
                            audio_np,
                            orig_sr=source.SAMPLE_RATE,
                            target_sr=16000
                        )
                    except Exception as e:
                        print(f"Error resampling audio: {str(e)}")
                        audio_resampled = audio_np
                   
                    text = WHISPER_PIPELINE(
                        audio_resampled,
                        generate_kwargs={"task": "transcribe", "language": "en"},
                        return_timestamps=True
                    )["text"].strip()
                    
                    processing_end_time = time.time()
                   
                    if not text:
                        print("No text transcribed from audio")
                        text = ""
                       
                    print(f"You said: {text}")
                   
                    exit_commands = ["quit", "end", "stop", "exit", "end interview", "exit interview", "stop interview", "quit interview"]
                    if text and any(cmd == text.lower().strip() for cmd in exit_commands):
                        return "quit", audio_np, processing_end_time
                   
                    if not text or len(text.split()) < 3:
                        if current_retry < max_retries - 1:
                            prompt = "I didn't quite catch that. Could you please say that again?"
                            print(prompt)
                            speak(prompt)
                            current_retry += 1
                            continue
                       
                        return "", np.array([], dtype=np.float32), processing_end_time
                    else:
                        return text, audio_np, processing_end_time
                       
                except Exception as e:
                    print(f"Error in Whisper speech recognition: {str(e)}")
                    if current_retry < max_retries - 1:
                        prompt = "I'm having trouble understanding. Could you please try again?"
                        print(prompt)
                        speak(prompt)
                        current_retry += 1
                        continue
                   
                    return "", np.array([], dtype=np.float32), time.time()
                   
        except Exception as e:
            print(f"Error processing audio frames: {str(e)}")
            if current_retry < max_retries - 1:
                prompt = "There was a problem processing your response. Could you try again?"
                print(prompt)
                speak(prompt)
                current_retry += 1
                continue
           
            return "", np.array([], dtype=np.float32), time.time()
    
    return "", np.array([], dtype=np.float32), time.time()
   

def listen_and_save_name(recording_dir, question_id, silence_threshold=3, min_speech_duration=1, max_recording_duration=180):
    max_retries = 1
    current_retry = 0

    mic_device_index = 2  # Setting device index to 2 as requested
   
    while current_retry < max_retries:
        recognizer = sr.Recognizer()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
       
        print(f"Starting recording attempt {current_retry+1}/{max_retries}...")
       
        try:
            # Add a short delay before opening the microphone
            time.sleep(0.5)
           
            # Specify device_index=2 when creating the microphone instance
            with sr.Microphone(device_index=mic_device_index) as source:
                print("Adjusting for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                print(f"Recording started. Speak now...")
               
                # Rest of the function remains unchanged
                recognizer.dynamic_energy_threshold = False
                fixed_threshold = 2000
                recognizer.energy_threshold = fixed_threshold

                audio_frames = []
                frame_duration = 0.1
                start_time = time.time()
                last_sound_time = start_time
               
                try:
                    while True:
                        current_time = time.time()
                        total_duration = current_time - start_time
                        silence_duration = current_time - last_sound_time
                       
                        print(f"\rRecording: {total_duration:.1f}s | Silence: {silence_duration:.1f}s/{silence_threshold}s", end="", flush=True)
                       
                        if total_duration >= max_recording_duration:
                            print("\nMaximum recording duration reached!")
                            break
                       
                        if silence_duration >= silence_threshold and total_duration > min_speech_duration:
                            print("\nSilence threshold reached!")
                            break
                       
                        try:
                            # Add a small timeout to avoid hardware blocking
                            frame = recognizer.record(source, duration=frame_duration)
                            audio_frames.append(frame)
                           
                            frame_data = frame.get_raw_data()
                            if len(frame_data) > 0:  # Make sure we have data
                                energy = sum(int.from_bytes(frame_data[i:i+2], byteorder='little', signed=True)**2
                                            for i in range(0, len(frame_data), 2)) / (len(frame_data)/2)
                               
                                if energy > fixed_threshold:
                                    last_sound_time = current_time
                           
                        except sr.WaitTimeoutError:
                            print("\nTimeout waiting for audio")
                            continue
                        except Exception as e:
                            print(f"\nError during frame recording: {str(e)}")
                            # Don't break immediately, try again after short delay
                            time.sleep(0.2)
                            continue
               
                except KeyboardInterrupt:
                    print("\nRecording interrupted by user")
               
                print("\nProcessing recorded audio...")
               
                if len(audio_frames) > 0:
                    # Safely combine audio frames
                    try:
                        combined_audio_data = b''.join(frame.get_raw_data() for frame in audio_frames)
                        combined_audio = sr.AudioData(combined_audio_data, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                       
                        audio_np = np.frombuffer(combined_audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                       
                        if recording_dir:
                            filename = f"{timestamp}_{question_id}_attempt{current_retry+1}.wav"
                            filepath = recording_dir / filename
                            try:
                                with open(filepath, "wb") as f:
                                    f.write(combined_audio.get_wav_data())
                                print(f"Audio saved to {filepath}")
                            except Exception as e:
                                print(f"Error saving audio file: {str(e)}")
                       
                        try:
                            text = recognizer.recognize_google(combined_audio)
                            print(f"You said: {text}")
                           
                            exit_commands = ["quit", "end", "stop", "exit", "end interview", "exit interview", "stop interview", "quit interview"]
                            if text and any(command == text.lower().strip() for command in exit_commands):
                                return "quit", audio_np
                           
                            if not text:
                                print("Please Answer Again.")
                                speak("Please Answer Again. ")
                                current_retry += 1
                                if current_retry >= max_retries:
                                    return "", np.array([], dtype=np.float32)
                            else:
                                return text, audio_np
                               
                        except sr.UnknownValueError:
                            print("Speech wasn't recognized.")
                            current_retry += 1
                            if current_retry >= max_retries:
                                return "", np.array([], dtype=np.float32)
                        except Exception as e:
                            print(f"Error in speech recognition: {str(e)}")
                            current_retry += 1
                            if current_retry >= max_retries:
                                return "", np.array([], dtype=np.float32)
                    except Exception as e:
                        print(f"Error processing audio frames: {str(e)}")
                        current_retry += 1
                        if current_retry >= max_retries:
                            return "", np.array([], dtype=np.float32)
                else:
                    print("No audio recorded.")
                    current_retry += 1
                    if current_retry >= max_retries:
                        return "", np.array([], dtype=np.float32)
                   
        except Exception as e:
            print(f"\nError during recording: {str(e)}")
            # Add a delay before retrying to let any system resources release
            time.sleep(1.0)
            current_retry += 1
            if current_retry >= max_retries:
                print("Moving to the next question due to technical issues.")
                return "", np.array([], dtype=np.float32)
   
    return "", np.array([], dtype=np.float32)
