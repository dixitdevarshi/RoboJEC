def conduct_interview(system, user_name, recording_dir):
    """Conduct the personality interview with proper question sequencing and time gap tracking"""
    welcome = f"\nGreat, {user_name}! Let's begin our conversation about your personality and experiences."
    print(welcome)
    speak(welcome)
    instructions = "I'll ask you questions to learn more about you. I'd love to hear your thoughts in as much detail (or as little) as you'd like."
    print(instructions)
    speak(instructions)

    # Create response directory
    response_dir = Path(f"./personality_responses/{user_name.lower().replace(' ', '_')}")
    response_dir.mkdir(parents=True, exist_ok=True)

    # Initialize tracking variables
    start_time = time.time()
    question_count = 0
    hobby_question_count = 0
    willingness_level = WillingnessLevel.MEDIUM
    willingness_score = 50
    last_question_was_followup = False

    # Time tracking variables
    professional_time_gaps = []  # For Phase 1 only
    hobby_time_gaps = []         # For Phase 3 only
    all_time_gaps = []           # Track all time gaps for accurate statistics
    
    last_response_end_time = None
    question_timings = []        # Track detailed timing information

    # Question sequence (4 subcategory, 2 main)
    category_sequence = ['subcategory', 'main', 'subcategory', 'subcategory', 'main', 'subcategory']
    main_subcategory_count = 0

    try:
        # ===== PHASE 1: Professional Questions =====
        print("\n=== PHASE 1: Professional Experience ===")
        while main_subcategory_count < 6:
            # Get next question
            current_category = category_sequence[main_subcategory_count]
            question = system.get_question_by_category(current_category, willingness_level)
            question_count += 1
            main_subcategory_count += 1

            # Track question start time (preparation time)
            question_prep_time = time.time()
            question_timings.append(("question_prep", question_prep_time))

            # Calculate and display time gap if applicable
            if last_response_end_time is not None:
                prep_gap = question_prep_time - last_response_end_time
                gap_msg = f"\n⏱️ Time between response and question prep: {prep_gap:.2f} seconds"
                print(gap_msg)  # Don't speak this to avoid interrupting flow

            # Display question info
            print(f"\nQuestion {question_count}/{len(category_sequence)}")
            print(f"Category: {question['category_name']}")
            print(f" {question['question_text']}")

            # Record and speak the question - now returns first byte timing too
            question_audio_file, first_byte_time = speak_and_record(
                question['question_text'], 
                recording_dir, 
                f"q{question_count}_{question['category_type']}_question"
            )
            
            # Calculate the actual gap to first byte of speech
            if last_response_end_time is not None:
                true_gap = first_byte_time - last_response_end_time
                professional_time_gaps.append(true_gap)  # Store in professional gaps
                all_time_gaps.append(true_gap)           # Also store in all gaps
                print(f"\n⏱️ Time from last response to first speech byte: {true_gap:.2f} seconds")
                
                # Record this timing data
                question_timings.append(("first_byte", first_byte_time))
                question_timings.append(("true_gap", true_gap))

            # Get response (now expects 3 return values)
            print("\n[Please respond now]")
            answer, audio_data, processing_end_time = listen_and_save(
                recording_dir=recording_dir,
                question_id=f"q{question_count}_{question['category_type']}",
                silence_threshold=3.5,
                min_speech_duration=1,
                max_recording_duration=480
            )
            
            # Use processing_end_time which includes transcription time
            last_response_end_time = processing_end_time
            question_timings.append(("response_end", last_response_end_time))

            # Check for early termination
            if answer and answer.lower() == 'quit':
                end_msg = "\nThank you for sharing! Interview ended at your request."
                print(end_msg)
                speak(end_msg)
                return

            # Process response
            full_answer = answer if answer else "[No response]"
            
            # Check for follow-up conditions
            if (answer and len(answer.split()) > 18 and not last_question_was_followup and random.random() < 0.6):
                keywords = extract_keywords(answer)
                themes = identify_themes(answer)
                missing_star = check_star(answer)
                substantial_response = (
                    len(keywords) >= 3 or 
                    any(theme in themes for theme in ["teamwork", "leadership", "challenge", "achievement"]) or 
                    len(missing_star) >= 2
                )
                
                if substantial_response:
                    follow_up = system.followup_generator.generate_follow_up(
                        answer, keywords, len(answer.split()), False, themes, missing_star
                    )
                    
                    if follow_up:
                        print(f"\n[Follow-up] {follow_up}")
                        
                        # Track follow-up preparation time
                        followup_prep_time = time.time()
                        question_timings.append(("followup_prep", followup_prep_time))
                        
                        # Calculate gap for follow-up preparation
                        if last_response_end_time is not None:
                            followup_prep_gap = followup_prep_time - last_response_end_time
                            print(f"⏱️ Time between response and follow-up prep: {followup_prep_gap:.2f} seconds")
                        
                        # Record and speak follow-up - now capturing first byte timing
                        followup_audio_file, followup_first_byte = speak_and_record(
                            follow_up, 
                            recording_dir, 
                            f"q{question_count}_followup_question"
                        )
                        
                        # Calculate the true gap to first byte of follow-up speech
                        if last_response_end_time is not None:
                            followup_true_gap = followup_first_byte - last_response_end_time
                            professional_time_gaps.append(followup_true_gap)
                            all_time_gaps.append(followup_true_gap)
                            print(f"\n⏱️ Time from response to first follow-up speech byte: {followup_true_gap:.2f} seconds")
                            
                            # Record this timing data
                            question_timings.append(("followup_first_byte", followup_first_byte))
                            question_timings.append(("followup_true_gap", followup_true_gap))

                        # Get follow-up response (expects 3 return values)
                        print("\n[Please respond to follow-up]")
                        followup_answer, followup_audio, followup_end_time = listen_and_save(
                            recording_dir=recording_dir,
                            question_id=f"q{question_count}_followup",
                            silence_threshold=3.5,
                            min_speech_duration=1,
                            max_recording_duration=480
                        )
                        
                        # Track follow-up end time (including transcription)
                        last_response_end_time = followup_end_time
                        question_timings.append(("followup_end", last_response_end_time))

                        if followup_answer and followup_answer.lower() == 'quit':
                            end_msg = "\nThank you for sharing! Interview ended at your request."
                            print(end_msg)
                            speak(end_msg)
                            return

                        if followup_answer:
                            full_answer += f"\n\n[Follow-up] {follow_up}\n{followup_answer}"
                        
                        last_question_was_followup = True
                    else:
                        last_question_was_followup = False
                else:
                    last_question_was_followup = False
            else:
                last_question_was_followup = False

            # Save response
            save_response(
                response_dir,
                question,
                full_answer,
                recording_dir,
                f"q{question_count}_{question['category_type']}",
                audio_data,
                question_audio_file
            )
            
            # Update willingness level
            if audio_data is not None and len(audio_data) > 0:
                new_willingness, score = system.update_willingness_level(audio_data)
                willingness_level = new_willingness
                willingness_score = score

        # ===== PHASE 2: Hobby Discovery =====
        print("\n=== PHASE 2: Personal Interests ===")
        # Reset time tracking to create a break between phases
        # We don't calculate a gap between Phase 1 and Phase 2
        last_phase1_end_time = last_response_end_time
        last_response_end_time = None  # Reset to avoid calculating gap to Phase 2
        
        # Mark start of transition statement
        transition_start = time.time()
        transition = "Now I'd like to learn about your interests."
        print(f"\nQuestion {question_count + 1} [Hobby Discovery]")
        print(f" {transition}")
        
        # Get first byte timing for transition
        transition_audio_file, transition_first_byte = speak_and_record(
            transition,
            recording_dir,
            f"hobby_transition"
        )
        question_timings.append(("hobby_transition_first_byte", transition_first_byte))
        
        # For hobby question timing, we'll track when we start processing the hobby question
        hobby_question_start = time.time()
        
        # Ask hobby question
        hobby_question = system.conversation_starter.generate_hobby_discovery_question(user_name)
        print(f"\nQuestion {question_count + 1} [Hobbies]")
        print(f" {hobby_question}")
        
        # Record and speak hobby question with first byte timing
        hobby_question_audio_file, hobby_q_first_byte = speak_and_record(
            hobby_question,
            recording_dir,
            f"hobby_discovery_question"
        )
        question_timings.append(("hobby_discovery_first_byte", hobby_q_first_byte))
        
        # Get hobby response (expects 3 return values)
        print("\n[Please share your interests]")
        hobby_answer, hobby_audio, hobby_end_time = listen_and_save(
            recording_dir=recording_dir,
            question_id=f"hobby_discovery",
            silence_threshold=3.5,
            min_speech_duration=1,
            max_recording_duration=480
        )
        
        # Track hobby response end time (including transcription)
        last_response_end_time = hobby_end_time
        question_timings.append(("hobby_response_end", last_response_end_time))
        
        # Check for early termination
        if hobby_answer and hobby_answer.lower() == 'quit':
            end_msg = "\nThank you for sharing! Interview ended at your request."
            print(end_msg)
            speak(end_msg)
            return
            
        # Process hobbies
        hobbies = extract_hobbies(hobby_answer) if hobby_answer else []
        
        # Save hobby response
        hobby_question_obj = {
            "category_type": "hobby_discovery",
            "category_name": "Personal Interests",
            "question_text": hobby_question,
            "willingness_level": str(willingness_level).lower()
        }
        save_response(
            response_dir,
            hobby_question_obj,
            hobby_answer if hobby_answer else "[No response]",
            recording_dir,
            f"hobby_discovery",
            hobby_audio,
            hobby_question_audio_file
        )
        
        # ===== PHASE 3: Hobby Questions =====
        if hobbies:
            print("\n=== PHASE 3: Hobby Questions ===")
            selected_hobby = hobbies[0]
            
            # Mark start of hobby introduction
            hobby_intro_start = time.time()
            hobby_intro = f"I hear that you enjoy {', '.join(hobbies)}. I'll focus on your interest in {selected_hobby}."
            print(hobby_intro)
            
            # Get first byte timing for hobby intro
            hobby_intro_audio, hobby_intro_first_byte = speak_and_record(
                hobby_intro,
                recording_dir,
                f"hobby_intro"
            )
            question_timings.append(("hobby_intro_first_byte", hobby_intro_first_byte))
            
            # Update system's hobby category
            system.categories["hobby"] = selected_hobby if selected_hobby else ""
            
            # Generate hobby dataset in background
            dataset_ready = threading.Event()
            print("\nGenerating questions about your hobby...")
            generation_thread = threading.Thread(
                target=generate_dataset_background,
                args=(system.question_generator, "hobby", selected_hobby, system, dataset_ready)
            )
            generation_thread.daemon = True
            generation_thread.start()
            
            # Reset time tracking for hobby phase - fresh start
            # Set to None initially so no gap is calculated for the first question
            hobby_last_response_end_time = None
            
            # Ask 3 hobby questions
            for i in range(3):
                question_count += 1
                hobby_question_count += 1
                
                # Start timing for hobby question preparation
                hobby_q_prep_time = time.time()
                question_timings.append((f"hobby_q{i+1}_prep", hobby_q_prep_time))
                
                # Calculate gap before each hobby question, but only after first one 
                # (i.e., start tracking gaps after the first hobby question response)
                if i > 0 and hobby_last_response_end_time is not None:
                    prep_time_gap = hobby_q_prep_time - hobby_last_response_end_time
                    hobby_time_gaps.append(prep_time_gap)  # Store in hobby gaps
                    all_time_gaps.append(prep_time_gap)    # Also store in all gaps
                    print(f"\n⏱️ Time between response and hobby question prep: {prep_time_gap:.2f} seconds")
                    question_timings.append((f"hobby_q{i+1}_prep_gap", prep_time_gap))
                elif i == 0:
                    # For the first hobby question, just note we're starting a new sequence
                    print("\n⏱️ Starting new timing sequence for hobby questions")
                
                # Get appropriate question
                if i == 0:
                    default_questions = system.question_generator._get_default_questions(
                        "hobby", selected_hobby, num_questions=1
                    )
                    question = (
                        system.create_question_from_template(default_questions[0])
                        if default_questions
                        else system._get_default_question(willingness_level)
                    )
                else:
                    if i == 1:
                        dataset_ready.wait(timeout=3)
                        if dataset_ready.is_set():
                            print("Using generated hobby questions.")
                            system.update_available_datasets("hobby")
                    question = system.get_question_by_category("hobby", willingness_level)
                
                # Display question
                print(f"\nQuestion {question_count} [Hobby {hobby_question_count}/3]")
                print(f"Category: {selected_hobby}")
                print(f" {question['question_text']}")
                
                # Record and speak the hobby question with first byte timing
                hobby_q_audio_file, hobby_q_first_byte = speak_and_record(
                    question['question_text'],
                    recording_dir,
                    f"q{question_count}_hobby_question"
                )
                question_timings.append((f"hobby_q{i+1}_first_byte", hobby_q_first_byte))
                
                # Calculate the true gap to first byte of speech for hobby questions
                # Only do this for questions after the first one
                if i > 0 and hobby_last_response_end_time is not None:
                    hobby_true_gap = hobby_q_first_byte - hobby_last_response_end_time
                    hobby_time_gaps.append(hobby_true_gap)
                    all_time_gaps.append(hobby_true_gap)
                    print(f"\n⏱️ Time from last response to first hobby speech byte: {hobby_true_gap:.2f} seconds")
                    question_timings.append((f"hobby_q{i+1}_true_gap", hobby_true_gap))
                
                # Get response (expects 3 return values)
                print("\n[Your thoughts]")
                answer, audio_data, processing_end_time = listen_and_save(
                    recording_dir=recording_dir,
                    question_id=f"q{question_count}_hobby",
                    silence_threshold=3.5,
                    min_speech_duration=1,
                    max_recording_duration=480
                )
                
                # Update the hobby-specific response end time tracker
                hobby_last_response_end_time = processing_end_time
                # Also update the global tracker for other functionality
                last_response_end_time = processing_end_time
                question_timings.append((f"hobby_q{i+1}_response_end", last_response_end_time))
                
                # Check for early termination
                if answer and answer.lower() == 'quit':
                    end_msg = "\nThank you for sharing! Interview ended at your request."
                    print(end_msg)
                    speak(end_msg)
                    return
                    
                # Save response
                save_response(
                    response_dir,
                    question,
                    answer if answer else "[No response]",
                    recording_dir,
                    f"q{question_count}_hobby",
                    audio_data,
                    hobby_q_audio_file
                )
                
                # Update willingness
                if audio_data is not None and len(audio_data) > 0:
                    new_willingness, score = system.update_willingness_level(audio_data)
                    willingness_level = new_willingness
                    willingness_score = score
        
        # ===== CONCLUSION =====
        elapsed = time.time() - start_time
        print(f"\nInterview complete! Duration: {int(elapsed//60)}m {int(elapsed%60)}s")
        print(f"Total questions: {question_count}")
        
        # Calculate and display time statistics using all captured time gaps
        if all_time_gaps:
            # Overall statistics from the comprehensive list
            avg_gap = statistics.mean(all_time_gaps)
            max_gap = max(all_time_gaps)
            min_gap = min(all_time_gaps)
            print(f"\n⏱️ Overall Time Gap Statistics:")
            print(f"- Average: {avg_gap:.2f} seconds")
            print(f"- Longest: {max_gap:.2f} seconds")
            print(f"- Shortest: {min_gap:.2f} seconds")
            
            # Professional phase statistics
            if professional_time_gaps:
                prof_avg = statistics.mean(professional_time_gaps)
                prof_max = max(professional_time_gaps)
                prof_min = min(professional_time_gaps)
                print(f"\n⏱️ Professional Phase Time Gaps:")
                print(f"- Average: {prof_avg:.2f} seconds")
                print(f"- Longest: {prof_max:.2f} seconds")
                print(f"- Shortest: {prof_min:.2f} seconds")
                
            # Hobby phase statistics
            if hobby_time_gaps:
                hobby_avg = statistics.mean(hobby_time_gaps)
                hobby_max = max(hobby_time_gaps)
                hobby_min = min(hobby_time_gaps)
                print(f"\n⏱️ Hobby Phase Time Gaps:")
                print(f"- Average: {hobby_avg:.2f} seconds")
                print(f"- Longest: {hobby_max:.2f} seconds")
                print(f"- Shortest: {hobby_min:.2f} seconds")
        
        # Save timing data to file with enhanced detail
        timing_file = response_dir / "timing_data.csv"
        with open(timing_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header row
            writer.writerow(["Phase", "Event Type", "Event Name", "Timestamp", "Gap (seconds)"])
            
            # Write all detailed timing events
            for event_name, timestamp in question_timings:
                # Determine phase based on event name
                if "hobby" in event_name:
                    if "discovery" in event_name:
                        phase = "Hobby Discovery"
                    else:
                        phase = "Hobby Questions"
                else:
                    phase = "Professional"
                
                # Determine event type
                if "prep" in event_name:
                    event_type = "Question Preparation"
                elif "first_byte" in event_name:
                    event_type = "First Speech Byte"
                elif "response_end" in event_name:
                    event_type = "Response Completion"
                elif "true_gap" in event_name:
                    event_type = "Response-to-Speech Gap"
                elif "intro" in event_name:
                    event_type = "Introduction"
                else:
                    event_type = "Other"
                
                # Write the row, formatting timestamp to 3 decimal places
                writer.writerow([phase, event_type, event_name, f"{timestamp:.3f}", 
                                ("" if "gap" not in event_name else f"{timestamp:.3f}")])
            
            # Write summary statistics
            writer.writerow(["", "", "", "", ""])
            writer.writerow(["Summary Statistics", "", "", "", ""])
            
            # Overall gap statistics
            if all_time_gaps:
                writer.writerow(["All Phases", "Average Gap", f"{statistics.mean(all_time_gaps):.3f}", "", ""])
                writer.writerow(["All Phases", "Max Gap", f"{max(all_time_gaps):.3f}", "", ""])
                writer.writerow(["All Phases", "Min Gap", f"{min(all_time_gaps):.3f}", "", ""])
                writer.writerow(["All Phases", "StdDev", f"{statistics.stdev(all_time_gaps) if len(all_time_gaps) > 1 else 0:.3f}", "", ""])
            
            # Professional phase statistics
            if professional_time_gaps:
                writer.writerow(["Professional", "Average Gap", f"{statistics.mean(professional_time_gaps):.3f}", "", ""])
                writer.writerow(["Professional", "Max Gap", f"{max(professional_time_gaps):.3f}", "", ""])
                writer.writerow(["Professional", "Min Gap", f"{min(professional_time_gaps):.3f}", "", ""])
                writer.writerow(["Professional", "StdDev", f"{statistics.stdev(professional_time_gaps) if len(professional_time_gaps) > 1 else 0:.3f}", "", ""])
            
            # Hobby phase statistics
            if hobby_time_gaps:
                writer.writerow(["Hobby", "Average Gap", f"{statistics.mean(hobby_time_gaps):.3f}", "", ""])
                writer.writerow(["Hobby", "Max Gap", f"{max(hobby_time_gaps):.3f}", "", ""])
                writer.writerow(["Hobby", "Min Gap", f"{min(hobby_time_gaps):.3f}", "", ""])
                writer.writerow(["Hobby", "StdDev", f"{statistics.stdev(hobby_time_gaps) if len(hobby_time_gaps) > 1 else 0:.3f}", "", ""])
            
            # Write gap data in raw form for further analysis
            writer.writerow(["", "", "", "", ""])
            writer.writerow(["Raw Gap Data", "", "", "", ""])
            writer.writerow(["Phase", "Gap #", "Seconds", "", ""])
            
            # Professional gaps
            for i, gap in enumerate(professional_time_gaps, 1):
                writer.writerow(["Professional", f"P{i}", f"{gap:.3f}", "", ""])
            
            # Hobby gaps
            for i, gap in enumerate(hobby_time_gaps, 1):
                writer.writerow(["Hobby", f"H{i}", f"{gap:.3f}", "", ""])
        
        # Final feedback question
        ask_feedback_question(user_name, response_dir, recording_dir)
        
        # Closing messages
        closing_msg = (f"\nThank you for giving your valuable time, {user_name}. I truly appreciate it")
        print(closing_msg)
        speak(closing_msg)
        time.sleep(1)
        
        audi = (f"\nAnd last but definitely not least, thank you to our wonderful audience for being here. "
                f"From all of us at RoboJec team, Dr. Agya Mishra, Devarshi Dixit, and Sanskriti Jain,—"
                f"we're so glad you joined us. Have a great day!")
        print(audi)
        speak(audi)
    
    except KeyboardInterrupt:
        print("\nInterview interrupted")
        elapsed = time.time() - start_time
        print(f"Completed {question_count} questions in {int(elapsed//60)}m {int(elapsed%60)}s")
        
        # Calculate average time gaps even if interrupted
        if all_time_gaps:
            avg_gap = statistics.mean(all_time_gaps)
            max_gap = max(all_time_gaps)
            min_gap = min(all_time_gaps)
            print(f"\n⏱️ Overall Time Gap Statistics:")
            print(f"- Average: {avg_gap:.2f} seconds")
            print(f"- Longest: {max_gap:.2f} seconds")
            print(f"- Shortest: {min_gap:.2f} seconds")
            
        # Save whatever timing data we have collected so far
        timing_file = response_dir / "timing_data_interrupted.csv"
        with open(timing_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Event", "Timestamp", "Notes"])
            for event_name, timestamp in question_timings:
                writer.writerow([event_name, f"{timestamp:.3f}", "Interrupted interview"])


def ask_feedback_question(user_name, response_dir, recording_dir):
    """Ask a single feedback question after the interview is complete."""
    # List of possible feedback questions
    feedback_questions = [
        f"Before we conclude, {user_name}, what's one piece of advice you'd give to young people?",
        f"One last question, If you could plant one piece of wisdom in the minds of young people today, what would it be?"
    ]
   
    # Pick a random feedback question
    feedback_q = random.choice(feedback_questions)
   
    # Ask the feedback question
    print(feedback_q)
    
    # Record and speak the feedback question
    feedback_q_audio_file = speak_and_record(
        feedback_q,
        recording_dir,
        "feedback_question"
    )
   
    # Get user's feedback
    print("\nListening for your answer...")
    feedback_text, feedback_audio = listen_and_save_name(
        recording_dir=recording_dir,
        question_id="feedback",
        silence_threshold=3.0,  # Allow more time for thoughtful feedback
        min_speech_duration=1.5,
        max_recording_duration=480  # 2 minutes max for feedback
    )
   
    # Create a feedback question object
    feedback_question = {
        "category_type": "feedback",
        "category_name": "Interview Experience",
        "question_text": feedback_q,
        "willingness_level": "medium"
    }
   
    # Save the feedback with question audio
    if feedback_text and feedback_text.lower() != 'quit':
        time_spent = save_response(
            response_dir,
            feedback_question,
            feedback_text,
            recording_dir,
            "feedback",
            feedback_audio,
            feedback_q_audio_file
        )
       
def save_response(response_dir, question, answer, recording_dir, recording_id, audio_data=None, question_audio_file=None):
    """
    Save interview response with precise timing information.
    Now also includes reference to the question audio file.
    """
    try:
        # Calculate word count with safety check
        word_count = len(answer.split()) if answer else 0
       
        # Estimate time based on word count (fallback)
        estimated_time = word_count / 2.5 if word_count > 0 else 1.0
       
        precise_time = None
        # Get time from audio data if available
        if audio_data is not None and hasattr(audio_data, 'size') and audio_data.size > 0:
            # Ensure sample rate is positive
            sample_rate = 16000  # Default sample rate
            precise_time = len(audio_data) / sample_rate
       
        file_time = None
        # Try to get time from audio file
        audio_file = recording_dir / f"{recording_id}.wav"
        if audio_file.exists():
            try:
                import wave
                with wave.open(str(audio_file), 'rb') as wf:
                    frames = wf.getnframes()
                    rate = float(wf.getframerate())
                    if rate > 0:  # Safety check for division
                        file_time = frames / rate
            except Exception as e:
                print(f"Warning: Couldn't read audio duration: {str(e)}")
       
        # Use the best available time measure
        time_spent = precise_time or file_time or estimated_time
       
        # Ensure we have a positive time value
        time_spent = max(0.1, time_spent)
       
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
       
        response_data = {
            "timestamp": timestamp,
            "category_type": question.get('category_type', 'unknown'),
            "category_name": question.get('category_name', 'unknown'),
            "question_text": question.get('question_text', ''),
            "answer_text": answer,
            "word_count": word_count,
            "time_spent": round(time_spent, 2),
            "willingness_level": question.get('willingness_level', 'medium'),
            "audio_file": str(audio_file.relative_to(recording_dir.parent)) if audio_file.exists() else None,
            "audio_duration": round(file_time, 2) if file_time else round(time_spent, 2),
            "question_audio_file": str(question_audio_file.relative_to(recording_dir.parent)) if question_audio_file and question_audio_file.exists() else None
        }
       
        # Save to CSV
        response_file = response_dir / "interview_responses.csv"
        file_exists = response_file.exists()
       
        with open(response_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=response_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(response_data)
       
        # Save JSON version
        json_file = response_dir / f"{recording_id}_response.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(response_data, f, indent=2)
       
        return time_spent
    except Exception as e:
        print(f"Error saving response: {str(e)}")
        # Return a safe fallback value
        return estimated_time if 'estimated_time' in locals() else 1.0