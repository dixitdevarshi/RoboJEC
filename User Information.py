def get_user_info():
    """Collect basic information from the user using voice conversation."""
    print("\n--- Welcome to the Voice-Based Personality Interview System ---\n")
   
    # Initialize conversation starter
    starter = ConversationStarterGenerator()
   
    # Wait for 5 seconds for potential introduction
    print("\nListening for introduction... (waiting 5 seconds)")
    intro_response_text, intro_response_audio = listen_and_save_name(None, "initial_response", silence_threshold=5, min_speech_duration=1, max_recording_duration=5)
   
    # Extract name from response
    name = starter.extract_name(intro_response_text) if intro_response_text else None

    if name and name != "Friend":
        # If name was provided in introduction, use the AI introduction
        ai_intro = random.choice(starter.ai_introductions).format(name=name)
        ai_wel = random.choice(starter.welcome2).format(name=name)
        print(ai_intro)
        speak(ai_intro)
        time.sleep(2)
        print(ai_wel)
        speak(ai_wel)

        
        
    else:
        # If no name after waiting, introduce ourselves and ask
        followup = random.choice(starter.followups)
        print(followup)
        speak(followup)
       
        # Ask for name
        name_q = random.choice(starter.name_questions)
        print(name_q)
        speak(name_q)
       
        name_response_text, name_response_audio = listen_and_save_name(None, "name_prompt")
        name = starter.extract_name(name_response_text)
        while not name:
            prompt = "I didn't catch your name. Could you please tell me your name?"
            print(prompt)
            speak(prompt)
            response_text, response_audio = listen_and_save_name(None, "name_prompt")
            name = starter.extract_name(response_text)

        welcome_msg = random.choice(starter.welcome).format(name=name)
        ai_wel = random.choice(starter.welcome2).format(name=name)
        print(welcome_msg)
        speak(welcome_msg)
        time.sleep(2)
        print(ai_wel)
        speak(ai_wel)

   
    # Create user directory for recordings now that we have the name
    user_recording_dir = create_user_recording_directory(name)
   
    # Generate personalized questions
    prof_q, exp_q, hobby_q = starter.generate_personalized_questions(name)
   
    # Get profession and experience with improved validation
    valid_categories = False
    profession = ""
    years_experience = 0
    profession_categories = None
   
    while not valid_categories:
        # Ask about profession
        print(prof_q)
        speak(prof_q)
        profession_text, profession_audio = listen_and_save_name(user_recording_dir, "profession")
        profession = profession_text.strip().lower() if profession_text else ""
       
        if not profession:
            prompt = "Could you tell me about your profession?"
            print(prompt)
            speak(prompt)
            continue
       
        # Ask about experience
        print(exp_q)
        speak(exp_q)
       
        # Get years of experience with validation
        years_experience = None
        attempt = 1
        while years_experience is None:
            exp_text, exp_audio = listen_and_save_name(user_recording_dir, f"experience_{attempt}")
            years_experience = extract_number_from_text(exp_text)
           
            if years_experience is not None and years_experience >= 0:
                break
            elif years_experience is not None and years_experience < 0:
                prompt = "Experience cannot be negative. Please enter a valid number."
                print(prompt)
                speak(prompt)
                years_experience = None
            else:
                prompt = "Please say a number for your years of experience."
                print(prompt)
                speak(prompt)
                attempt += 1
       
        # Use the profession recognizer instead of map_profession_to_categories
        try:
            profession_categories = recognize_profession(profession, years_experience)
            if profession_categories:
                valid_categories = True
                # Provide feedback on the recognized profession
                recognition_info = f"Recognized as: {profession_categories['main_category']} - {profession_categories['subcategory']}"
                print(recognition_info)
            else:
                prompt = "I'm not familiar with that profession. Please specify a recognized profession using common industry terms."
                print(prompt)
                speak(prompt)
        except Exception as e:
            print(f"Error recognizing profession: {e}")
            prompt = "I'm having trouble recognizing that profession. Please try again."
            print(prompt)
            speak(prompt)
   
    # Print confirmation but don't speak it
    confirmation = f"\nThank you, {name}!"
    print(confirmation)
    info = f"Profession: {profession}, Years of Experience: {years_experience}"
    print(info)
   
    # Add profession category information
    category_info = f"Industry: {profession_categories['main_category']}, Role: {profession_categories['subcategory']}"
    if profession_categories.get('interdisciplinary', False):
        category_info += f", Secondary Industry: {profession_categories['secondary_category']}"
    print(category_info)
   
    preparing = "Preparing your personalized interview..."
    print(f"\n{preparing}\n")
   
    return {
        "name": name,
        "profession": profession,
        "profession_categories": profession_categories,  # Include the profession categories data
        "years_experience": years_experience,
        "hobbies": [],  # Return an empty list - hobbies will be determined during the interview
        "recording_dir": user_recording_dir,
    }
