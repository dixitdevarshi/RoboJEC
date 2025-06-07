class PersonalityInterviewSystem:
    def __init__(self, client, questions_dir: Path, main_category: str, subcategory: str, hobby: str):
        self.question_generator = PersonalityQuestionsGenerator(client, questions_dir)
        self.followup_generator = FollowUpGenerator(client) 
        # Store all categories for the system
        self.categories = {
            "main": main_category,
            "subcategory": subcategory,
            "hobby": hobby if hobby is not None else ""
        }
        self.asked_questions = set()
        self.available_datasets = self._check_available_datasets()
        self.all_questions = self._load_all_questions()
        self.lock = threading.Lock()
        self.willingness_analyzer = WillingnessAnalyzer()
        self.current_willingness = WillingnessLevel.MEDIUM
        # Add the missing conversation_starter attribute
        self.conversation_starter = ConversationStarterGenerator()
        # Track questions asked per category
        self.category_question_counts = {
            "main": 0,
            "subcategory": 0,
            "hobby": 0
        }

    def _check_available_datasets(self):
        """Check which datasets are available."""
        available = {}
        for category_type, category_name in self.categories.items():
            # Skip empty hobby
            if category_type == "hobby" and not category_name:
                available[category_type] = False
                continue
            file_path = self.question_generator._get_category_file_path(category_type, category_name)
            available[category_type] = file_path.exists()
        return available

    def refresh_available_datasets(self):
        """Refresh the available datasets lookup."""
        with self.lock:
            self.available_datasets = self._check_available_datasets()

    def _load_all_questions(self):
        """Load all available questions across all categories and willingness levels."""
        all_questions = set()
        # Try to load questions from each category
        for category_type, category_name in self.categories.items():
            # Skip if category is empty
            if not category_name:
                continue
           
            # Even if dataset isn't available yet, try to get default questions
            if not self.available_datasets.get(category_type, False):
                default_qs = self.question_generator._get_default_questions(
                    category_type=category_type,
                    category=category_name,
                    num_questions=10
                )
                all_questions.update({q["question"] for q in default_qs if "question" in q})
                continue
               
            # Load questions for each willingness level
            for willingness in WillingnessLevel:
                questions = self.question_generator.get_questions(
                    category_type=category_type,
                    category=category_name,
                    num_questions=25,
                    willingness_level=willingness
                )
                all_questions.update({q["question"] for q in questions if "question" in q})
        return all_questions

    def update_available_datasets(self, category_type):
        """Update the available datasets tracker when a new dataset is generated."""
        with self.lock:
            self.available_datasets[category_type] = True
            print(f"\nSystem updated: {category_type} dataset is now available.")

    def get_question_by_category(self, category_type, willingness_level=None, feedback=None):
        """Get a question specifically from the requested category."""
        if willingness_level is None:
            willingness_level = self.current_willingness
       
        category_name = self.categories[category_type]
       
        # Early return with default if category is empty
        if not category_name:
            return self._get_default_question(willingness_level, category_type)
       
        # Double-check if the dataset file exists on disk, even if our tracking says it doesn't
        if not self.available_datasets.get(category_type, False):
            file_path = self.question_generator._get_category_file_path(category_type, category_name)
            if file_path.exists():
                # Dataset exists but we weren't aware of it - update our tracking
                self.update_available_datasets(category_type)
                print(f"\nDetected newly available dataset for {category_type}: {category_name}")
       
        # If dataset still isn't available, use default question
        if not self.available_datasets.get(category_type, False):
            return self._get_default_question(willingness_level, category_type)
       
        # Try to get questions for the specified category with different willingness levels if needed
        all_levels = list(WillingnessLevel)
        start_index = all_levels.index(willingness_level)
       
        # Try each willingness level in turn
        for i in range(len(all_levels)):
            # Get the next willingness level to try (cycle through them)
            current_level = all_levels[(start_index + i) % len(all_levels)]
           
            # Get questions for this category and willingness level
            available_questions = self.question_generator.get_questions(
                category_type=category_type,
                category=category_name,
                num_questions=25,
                willingness_level=current_level
            )
           
            # Filter out questions we've already asked
            new_questions = [q for q in available_questions if q["question"] not in self.asked_questions]
           
            # If we have new questions, select one
            if new_questions and len(new_questions) > 0:
                selected_question = random.choice(new_questions)
                # Mark as asked
                self.asked_questions.add(selected_question["question"])
                # Update category count
                self.category_question_counts[category_type] += 1
                # Return formatted question with metadata
                return {
                    "question_text": selected_question["question"],
                    "category_type": category_type,
                    "category_name": category_name,
                    "willingness_level": selected_question["willingness_level"]
                }
       
        # If we've exhausted all willingness levels and still have no questions, use a default
        return self._get_default_question(willingness_level, category_type)

    def _get_default_question(self, willingness_level, category_type=None):
        """Get a default question when no dataset is available."""
        # Use specified category or random if not specified
        if category_type is None:
            category_type = random.choice(list(filter(
                lambda x: self.categories[x],  # Filter out empty categories
                self.categories.keys()
            )))
       
        category_name = self.categories[category_type]
       
        # Create a generic question if the category is empty
        if not category_name:
            return {
                "question_text": "Could you tell me more about your interests and experiences?",
                "category_type": category_type,
                "category_name": "general",
                "willingness_level": willingness_level.value
            }
           
        # Get default questions
        default_questions = self.question_generator._get_default_questions(
            category_type=category_type,
            category=category_name,
            num_questions=25
        )
       
        # Safety check - make sure we have default questions
        if not default_questions:
            # If no default questions are available, create a generic one
            return {
                "question_text": f"Could you tell me more about your experience with {category_name}?",
                "category_type": category_type,
                "category_name": category_name,
                "willingness_level": willingness_level.value
            }
           
        # Filter by willingness level matching
        matching_questions = [
            q for q in default_questions
            if q.get("willingness_level", "medium_willingness") == willingness_level.value
        ]
       
        # If no matching willingness, use any question
        if not matching_questions:
            matching_questions = default_questions
           
        # Filter out asked questions
        new_questions = [q for q in matching_questions if q["question"] not in self.asked_questions]
       
        # Select a question
        if new_questions:
            selected_question = random.choice(new_questions)
        else:
            # If all default questions used, reuse one
            selected_question = random.choice(matching_questions)
           
        self.asked_questions.add(selected_question["question"])
       
        # Update category count
        self.category_question_counts[category_type] += 1
       
        return {
            "question_text": selected_question["question"],
            "category_type": category_type,
            "category_name": category_name,
            "willingness_level": selected_question["willingness_level"]
        }

    def get_hobby_question(self, willingness_level=None):
        """Get a question specifically from the hobby category."""
        if willingness_level is None:
            willingness_level = self.current_willingness
           
        hobby = self.categories["hobby"]
       
        # Double-check if the hobby dataset exists on disk, even if our tracking says it doesn't
        if not self.available_datasets.get("hobby", False):
            file_path = self.question_generator._get_category_file_path("hobby", hobby)
            if file_path.exists():
                # Dataset exists but we weren't aware of it - update our tracking
                self.update_available_datasets("hobby")
                print(f"\nDetected newly available dataset for hobby: {hobby}")
       
        # Check if hobby dataset is available
        if not self.available_datasets.get("hobby", False):
            return None
           
        # Get questions for hobby category
        available_questions = self.question_generator.get_questions(
            category_type="hobby",
            category=hobby,
            num_questions=25,
            willingness_level=willingness_level
        )
       
        # Filter out questions we've already asked
        new_questions = [q for q in available_questions if q["question"] not in self.asked_questions]
       
        # If we have new questions, select one
        if new_questions:
            selected_question = random.choice(new_questions)
            # Mark as asked
            self.asked_questions.add(selected_question["question"])
            # Update category count
            self.category_question_counts["hobby"] += 1
            # Return formatted question with metadata
            return {
                "question_text": selected_question["question"],
                "category_type": "hobby",
                "category_name": hobby,
                "willingness_level": selected_question["willingness_level"]
            }
        return None

    def create_question_from_template(self, question_template):
        """Create a question dictionary from a template."""
        category_type = question_template.get("category_type", "hobby")
        return {
            "question_text": question_template["question"],
            "category_type": category_type,
            "category_name": self.categories[category_type],
            "willingness_level": question_template.get("willingness_level", "medium_willingness")
        }

    def update_willingness_level(self, audio_data):
        """Analyze the audio data to determine willingness level."""
        if audio_data is None or len(audio_data) == 0:
            return WillingnessLevel.LOW, 0
           
        level, score, details = self.willingness_analyzer.analyze_audio_data(audio_data)
        self.current_willingness = level
        print(f"\nWillingness Analysis: {level.value.upper()} (Score: {score:.1f})")
        print(f"Details: Volume: {details.get('volume_score', 0):.1f}, Speech: {details.get('speech_score', 0):.1f}, Engagement: {details.get('engagement_score', 0):.1f}")
        return level, score  


def run_interview_with_name(api_key=None, name=None, user_recording_dir=None):
    """Run the personalized interview system with a pre-identified user."""
    # Welcome message before system starts - print only
    #welcome_message = "Now let's continue with your personalized interview."
    #print(welcome_message)
    #speak(welcome_message)
   
    # Get API key from parameter, environment variable, or ask user
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            api_key = input("Enter your Anthropic API key: ")
   
    # Initialize the Anthropic client
    client = Anthropic(api_key=api_key)
   
    # If we don't have a user recording directory yet, create one
    if not user_recording_dir:
        user_recording_dir = create_user_recording_directory(name)
   
    # Get user's profession and experience directly
    # Initialize profession recognizer
    valid_categories = False
    profession = ""
    years_experience = 0
    profession_categories = None
   
    # Initialize the starter for personalized questions
    starter = ConversationStarterGenerator()
   
    # Generate personalized questions
    prof_q, exp_q, hobby_q = starter.generate_personalized_questions(name)
   
    # We already asked about profession in run_with_voice_activation, so wait for the response
    while not valid_categories:
        # Get profession response
        profession_text, profession_audio = listen_and_save(user_recording_dir, "profession")
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
            exp_text, exp_audio = listen_and_save(user_recording_dir, f"experience_{attempt}")
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
       
        # Use the profession recognizer
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
   
    # Collect user info into a dictionary
    user_info = {
        "name": name,
        "profession": profession,
        "profession_categories": profession_categories,
        "years_experience": years_experience,
        "hobbies": [],  # Return an empty list - hobbies will be determined during the interview
        "recording_dir": user_recording_dir,
    }
   
    # Create directory for this user's questions
    questions_dir = Path("./personality_questions")
    questions_dir.mkdir(parents=True, exist_ok=True)
   
    # Get profession categories
    main_category = profession_categories["main_category"]
    subcategory = profession_categories["subcategory"]
   
    # We'll initialize with no hobby and update it later
    hobby = None
   
    print("\nChecking for personalized question datasets...")
   
    # Initialize the question generator and check which datasets exist
    question_generator = PersonalityQuestionsGenerator(client, questions_dir)
   
    # Check if datasets exist and track which ones need to be generated
    datasets_status = {}
    datasets_to_generate = []
    for category_type, category_name in [
        ("main", main_category),
        ("subcategory", subcategory)
    ]:
        file_path = question_generator._get_category_file_path(category_type, category_name)
        if file_path.exists():
            print(f"✓ Found dataset for {category_type}: {category_name}")
            datasets_status[category_type] = True
        else:
            print(f"✗ Missing dataset for {category_type}: {category_name}")
            datasets_status[category_type] = False
            datasets_to_generate.append((category_type, category_name))

    # Don't check for hobby dataset yet, since we don't have a hobby
    datasets_status["hobby"] = False
   
    # Initialize the system with what we have
    print("\nStarting Personalized Interview:\n")
    print(f"Name: {user_info['name']}")
    print(f"Main category: {main_category}")
    print(f"Subcategory: {subcategory}")
    print(f"Hobby: To be determined during interview\n")
   
    if len(datasets_to_generate) > 0:
        print(f"Some datasets are missing. Starting interview with available questions while generating missing datasets in the background.")
   
    # Initialize the integrated system - don't pass hobby parameter at all
    system = PersonalityInterviewSystem(
        client=client,
        questions_dir=questions_dir,
        main_category=main_category,
        subcategory=subcategory,
        hobby=""
    )
   
    # Start generating missing datasets in separate threads
    generation_threads = []
    for category_type, category_name in datasets_to_generate:
        thread = threading.Thread(
            target=generate_dataset_background,
            args=(question_generator, category_type, category_name, system)
        )
        thread.daemon = True  # Background thread that won't block program exit
        thread.start()
        generation_threads.append(thread)
        print(f"Background generation started for {category_type}: {category_name}")
   
    # Start the interview
    conduct_interview(system, user_info["name"], user_info["recording_dir"])




def run_interview(api_key=None):
    """Run the personalized interview system with the user."""
     
    # Get API key from parameter, environment variable, or ask user
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            api_key = input("Enter your Anthropic API key: ")
   
    # Initialize the Anthropic client
    client = Anthropic(api_key=api_key)
   
    # Get user information (now includes recording directory)
    user_info = get_user_info()
   
    # Create directory for this user's questions
    questions_dir = Path("./personality_questions")
    questions_dir.mkdir(parents=True, exist_ok=True)
   
    # Get profession categories from user_info instead of calling map_profession_to_categories
    profession_categories = user_info["profession_categories"]
    main_category = profession_categories["main_category"]
    subcategory = profession_categories["subcategory"]
   
    # We'll initialize with no hobby and update it later
    hobby = None
   
    print("\nChecking for personalized question datasets...")
   
    # Initialize the question generator and check which datasets exist
    question_generator = PersonalityQuestionsGenerator(client, questions_dir)
   
    # Check if datasets exist and track which ones need to be generated
    datasets_status = {}
    # When checking datasets
    datasets_to_generate = []
    for category_type, category_name in [
        ("main", main_category),
        ("subcategory", subcategory)
    ]:
        file_path = question_generator._get_category_file_path(category_type, category_name)
        if file_path.exists():
            print(f"✓ Found dataset for {category_type}: {category_name}")
            datasets_status[category_type] = True
        else:
            print(f"✗ Missing dataset for {category_type}: {category_name}")
            datasets_status[category_type] = False
            datasets_to_generate.append((category_type, category_name))

    # Don't check for hobby dataset yet, since we don't have a hobby
    datasets_status["hobby"] = False
   
    # Initialize the system with what we have
    print("\nStarting Personalized Interview:\n")
    print(f"Name: {user_info['name']}")
    print(f"Main category: {main_category}")
    print(f"Subcategory: {subcategory}")
    print(f"Hobby: To be determined during interview\n")
   
    if len(datasets_to_generate) > 0:
        print(f"Some datasets are missing. Starting interview with available questions while generating missing datasets in the background.")
   
    # Initialize the integrated system - don't pass hobby parameter at all
    system = PersonalityInterviewSystem(
        client=client,
        questions_dir=questions_dir,
        main_category=main_category,
        subcategory=subcategory,
        hobby=""
    )
   
    # Start generating missing datasets in separate threads
    generation_threads = []
    for category_type, category_name in datasets_to_generate:
        thread = threading.Thread(
            target=generate_dataset_background,
            args=(question_generator, category_type, category_name, system)
        )
        thread.daemon = True  # Background thread that won't block program exit
        thread.start()
        generation_threads.append(thread)
        print(f"Background generation started for {category_type}: {category_name}")
   
    # Start the interview
    conduct_interview(system, user_info["name"], user_info["recording_dir"])