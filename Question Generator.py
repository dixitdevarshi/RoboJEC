class PersonalityQuestionsGenerator:
    def __init__(self, client, questions_dir: Path):
        self.client = client  
        self.questions_dir = questions_dir
       
        
        self.questions_dir.mkdir(parents=True, exist_ok=True)
        self.used_default_questions = set()
   
    def _get_default_questions(self, category_type: str, category: str, num_questions: int = 25) -> List[Dict[str, str]]:
        """Return default questions if no category-specific questions are available."""
   
        default_questions_templates = {
            "main": [
                f"How has {category} changed your daily routine?",
                f"What excites you most about {category} today?",
                f"Do you see {category} differently than five years ago?",
                f"What's the biggest challenge in {category} now?",
                f"Has {category} met your expectations so far?",
                f"Where do you see {category} heading next?",
                f"Is there something about {category} people misunderstand?",
                f"How has {category} impacted your career path?",
                f"What {category} trend seems overrated to you?",
                f"Do you think {category} is accessible to everyone?",
                f"What {category} skill seems most valuable today?",
                f"Has {category} changed how you solve problems?",
                f"What would improve {category} for beginners?",
                f"Do ethics in {category} get enough attention?",
                f"What's your favorite aspect of {category}?",
                f"Has {category} connected you with interesting people?",
                f"What's one {category} myth you'd like to debunk?",
                f"How do you stay current with {category}?",
                f"What {category} resource would you recommend?",
                f"Has {category} become more complex over time?",
                f"Do you think {category} is changing society positively?",
                f"What {category} development are you watching closely?",
                f"Is {category} headed in the right direction?",
                f"What drew you to {category} initially?",
                f"How might {category} evolve in five years?"
            ],
            "subcategory": [
                f"What's the best part of working in {category}?",
                f"Has your perspective on {category} changed over time?",
                f"What skill in {category} took longest to develop?",
                f"Do people misunderstand what {category} professionals actually do?",
                f"What {category} challenge do you face regularly?",
                f"Has technology changed how you approach {category}?",
                f"What attracted you to {category} initially?",
                f"Is work-life balance possible in {category}?",
                f"What {category} task do you find most rewarding?",
                f"Has {category} become more competitive recently?",
                f"What's something about {category} that surprised you?",
                f"Do you think {category} gets proper recognition?",
                f"What tool or method revolutionized your {category} work?",
                f"How do you explain {category} to someone unfamiliar?",
                f"What's changing fastest in {category} right now?",
                f"Do you mentor others in {category}?",
                f"What {category} skill is undervalued today?",
                f"Has your definition of success in {category} evolved?",
                f"What keeps you motivated in {category}?",
                f"Do you collaborate with others in {category}?",
                f"What would you change about {category} education?",
                f"Has specializing in {category} been worth it?",
                f"What's one {category} mistake people often make?",
                f"How do you handle stress in {category}?",
                f"What makes someone truly excel in {category}?"
            ],
            "hobby": [
                f"What first caught your interest in {category}?",
                f"What still amazes you about {category}?",
                f"Has {category} changed how you see things?",
                f"What's your personal style with {category}?",
                f"What {category} challenge changed you most?",
                f"What has {category} revealed about yourself?",
                f"What {category} experience would you share?",
                f"Has your approach to {category} evolved?",
                f"What {category} question still intrigues you?",
                f"Who has {category} connected you with?",
                f"Any personal rituals around your {category}?",
                f"Does your mood affect your {category} practice?",
                f"What's something hard about {category} for you?",
                f"Has {category} affected your relationships?",
                f"What aspect of {category} challenges you most?",
                f"How has {category} influenced your space?",
                f"What's your guiding principle with {category}?",
                f"Has {category} been healing for you?",
                f"What {category} myth have you disproven?",
                f"Has your background influenced your {category}?",
                f"Do you set limits around {category}?",
                f"Does perfectionism affect your {category}?",
                f"How would you map your {category} journey?",
                f"What values has {category} reinforced?",
                f"What's your most treasured {category} memory?"
            ]
        }
       
        questions_templates = default_questions_templates.get(category_type, default_questions_templates["main"])
       
        # Filter out questions that have already been used in this session
        unique_templates = []
        for template in questions_templates:
            question = template.format(category=category)
            if question not in self.used_default_questions:
                unique_templates.append(template)
   
        
        if not unique_templates:
            return []
   
        
        selected_templates = unique_templates[:num_questions]
   
        questions = []
        thirds = len(selected_templates) // 3
   
        for i, question_template in enumerate(selected_templates):
            question = question_template.format(category=category)
            # Add to used set
            self.used_default_questions.add(question)
       
            if i < thirds:
                willingness = "low_willingness"
            elif i < 2 * thirds:
                willingness = "medium_willingness"
            else:
                willingness = "high_willingness"
           
            questions.append({
                "question": question,
                "category_type": category_type,
                "category": category,
                "willingness_level": willingness
            })
   
        return questions


   
    def get_questions(self, category_type: str, category: str,
                      num_questions: int = 75,
                      willingness_level: WillingnessLevel = WillingnessLevel.MEDIUM) -> List[Dict[str, str]]:
        """Get questions for a specific category type and value."""

        # Try to load existing questions for this category
        file_path = self._get_category_file_path(category_type, category)
        if file_path.exists():
            with open(file_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                category_questions = list(reader)
                print(f"Loaded {len(category_questions)} existing questions for {category_type}: {category}")
        else:
            # Generate new questions if none exist
            print(f"No existing questions found for {category_type}: {category}, generating new ones...")
            category_questions = self._generate_category_questions(category_type, category, num_questions=75)  

        # Always check if we have questions before proceeding
        if not category_questions:
            # Return default questions if no category-specific questions were generated
            return self._get_default_questions(category_type, category, num_questions=num_questions)
   
        # Filter by willingness level
        available_questions = [
            q for q in category_questions
            if q.get("willingness_level", "medium_willingness") == willingness_level.value
        ]

        # If not enough questions for this willingness level, use any level
        if len(available_questions) < num_questions:
            available_questions = category_questions
   
        # Add safety check before random.sample
        if not available_questions:
            return self._get_default_questions(category_type, category, num_questions=num_questions)
   
        # Select random questions
        selected_questions = random.sample(
            available_questions,
            min(num_questions, len(available_questions))
        )

        return selected_questions
   
    def _generate_category_questions(self, category_type: str, category: str, num_questions: int = 75) -> List[Dict[str, str]]:
        """Generate questions specific to a category type and value."""
       
        prompts = {
            "main": f"""Generate {num_questions} unique conversational questions about the personality trait: {category}.

            REQUIREMENTS:
            - Questions MUST be 8-15 words each
            - Conversational, Sound like two personalities conversing
            - Sound natural like a podcast host, not an interviewer
            - Respectful but not overly sophisticated tone
            - Each question must explore a COMPLETELY DIFFERENT aspect of {category}
            - Avoid ANY semantic overlap between questions
            - Include personal touches like "How do you..." or "What's your..."
            - Be direct and to the point
            - NO lengthy setups or explanations
           
            Questions should explore different:
            - Life domains (work, relationships, personal)
            - Time periods (past, present, future)
            - Emotional contexts (challenges, joys, surprises)
            - Social contexts (alone, with others)
            - Aspects (benefits, challenges, evolution)
           
            FORMAT: One short question per line, no numbering.""",
           
            "subcategory": f"""Generate {num_questions} unique conversational questions about the profession: {category}.

            Context: {category} represents a professional role or career path within a broader industry.

            REQUIREMENTS:
            - Questions MUST be 8-15 words only
            - Sound like two personalities conversing, not interviewing
            - Respectful but not overly sophisticated tone
            - Questions should feel like one personality asking another
            - Direct and genuine, as if in a thoughtful dialogue
            - Each question explores a DIFFERENT aspect of working in {category}
            - No semantic overlap between questions
            - NO lengthy setups or explanations
           
            Questions should explore different:
            - Day-to-day realities of working in {category}
            - Career development and professional growth
            - Challenges and rewards of the profession
            - Skills and traits needed for success
            - Industry changes and adaptations
           
            BAD EXAMPLES (too formal/interview-like):
            - "What educational trajectory would you recommend for aspiring {category} professionals?"
            - "Could you delineate the primary challenges encountered in the {category} profession?"
            
            GOOD EXAMPLES (direct, conversational):
            - "What's the hardest part about being a {category}?"
            - "Has the {category} field changed since you started?"
            - "Do you think people understand what {category}s actually do?"
           
            FORMAT: One short question per line, no numbering.""",
           
            "hobby": f"""Generate {num_questions} unique conversational questions about the hobby: {category}.

            REQUIREMENTS:
            - Questions MUST be 8-15 words only
            - Sound like two personalities conversing, not interviewing
            - Respectful but not overly sophisticated tone
            - Questions should feel like one personality asking another
            - Direct and genuine, as if in a thoughtful dialogue
            - Each question explores a DIFFERENT aspect of {category}
            - No semantic overlap between questions
            - NO lengthy setups or explanations
           
            Questions should explore different:
            - Experience levels and skills
            - Personal discoveries and learning
            - Meaning and fulfillment aspects
            - Memorable moments and challenges
            - Social connections through the hobby
           
            BAD EXAMPLES (too formal/interview-like):
            - "What initially catalyzed your interest in pursuing {category} as a hobby?"
            - "How has your engagement with {category} facilitated personal development?"
            
            GOOD EXAMPLES (direct, conversational):
            - "What first drew you to {category}?"
            - "Has {category} changed something about you?"
            - "Do you share your {category} with others?"
           
            FORMAT: One short question per line, no numbering."""
        }

        try:
            prompt = prompts.get(category_type, prompts["main"])
           
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4000,  # Increased to allow for more questions
                messages=[{"role": "user", "content": prompt}]
            )
           
            # Extract and process questions
            questions = [
                line.strip() for line in response.content[0].text.split('\n')
                if line.strip() and '?' in line
            ]

            # Ensure we have at least the required number of unique questions
            unique_questions = list(set(questions))
            if len(unique_questions) < num_questions:
                # If not enough unique questions, make another API call with a modified prompt
                additional_prompt = f"""{prompt}

                IMPORTANT: Generate questions that are COMPLETELY DIFFERENT from these:

                {chr(10).join(unique_questions[:10])}

                REQUIREMENTS:
                - Questions MUST be 8-15 words only
                - Feel like one personality genuinely asking another
                - Direct, simple language a real person would use
                - Conversational, not academic or formal
                - Each question must explore a unique aspect of {category}
                - Avoid ANY words or phrases used in the examples above
                """
               
                additional_response = self.client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=4000,
                    messages=[{"role": "user", "content": additional_prompt}]
                )
               
                additional_questions = [
                    line.strip() for line in additional_response.content[0].text.split('\n')
                    if line.strip() and '?' in line
                ]
               
                # Filter out any semantically similar questions using basic keyword matching
                filtered_additional_questions = []
                for new_q in additional_questions:
                    # Convert to lowercase for comparison
                    new_q_lower = new_q.lower()
                   
                    # Flag for checking if the question is too similar to existing ones
                    too_similar = False
                   
                    for existing_q in unique_questions:
                        existing_q_lower = existing_q.lower()
                       
                        # Count matching significant words (words with length > 4)
                        significant_words_new = [w for w in new_q_lower.split() if len(w) > 4 and w not in ['about', 'would', 'could', 'think', 'there', 'their', 'where', 'when', 'what', 'that', 'have', 'your', 'with', 'this']]
                        significant_words_existing = [w for w in existing_q_lower.split() if len(w) > 4 and w not in ['about', 'would', 'could', 'think', 'there', 'their', 'where', 'when', 'what', 'that', 'have', 'your', 'with', 'this']]
                       
                        # Count matches
                        matches = sum(1 for w in significant_words_new if w in significant_words_existing)
                       
                        # If too many matches or structure is very similar, consider it too similar
                        if matches >= 3 or (new_q_lower.startswith(existing_q_lower[:15]) and len(existing_q_lower) > 20):
                            too_similar = True
                            break
                   
                    if not too_similar:
                        filtered_additional_questions.append(new_q)
               
                # Combine and deduplicate
                all_questions = list(set(unique_questions + filtered_additional_questions))
               
                # If we still don't have enough, try one more approach with an even more specific prompt
                if len(all_questions) < num_questions:
                    final_attempt_prompt = f"""Generate {num_questions - len(all_questions)} COMPLETELY UNIQUE questions about {category}.

                    REQUIREMENTS:
                    - Each question MUST be only 8-15 words
                    - Sound like one personality talking to another
                    - Direct and genuine, like in a real conversation
                    - Not formal, not sophisticated, just respectful
                    - Each must be COMPLETELY DIFFERENT from these examples:
                    {chr(10).join(all_questions[:15])}
                   
                    - Use different sentence structures and words
                    - Explore new dimensions not covered above
                    - NO setup or explanation text
                    
                    Remember: These should sound like one person naturally asking another about their experience with {category}.
                   
                    FORMAT: One short question per line, no numbering.
                    """
                   
                    final_response = self.client.messages.create(
                        model="claude-3-haiku-20240307",
                        max_tokens=4000,
                        messages=[{"role": "user", "content": final_attempt_prompt}]
                    )
                   
                    final_questions = [
                        line.strip() for line in final_response.content[0].text.split('\n')
                        if line.strip() and '?' in line
                    ]
                   
                    # Apply similar filtering
                    filtered_final_questions = []
                    for new_q in final_questions:
                        new_q_lower = new_q.lower()
                        too_similar = False
                       
                        for existing_q in all_questions:
                            existing_q_lower = existing_q.lower()
                            significant_words_new = [w for w in new_q_lower.split() if len(w) > 4 and w not in ['about', 'would', 'could', 'think', 'there', 'their', 'where', 'when', 'what', 'that', 'have', 'your', 'with', 'this']]
                            significant_words_existing = [w for w in existing_q_lower.split() if len(w) > 4 and w not in ['about', 'would', 'could', 'think', 'there', 'their', 'where', 'when', 'what', 'that', 'have', 'your', 'with', 'this']]
                            matches = sum(1 for w in significant_words_new if w in significant_words_existing)
                           
                            if matches >= 2:
                                too_similar = True
                                break
                       
                        if not too_similar:
                            filtered_final_questions.append(new_q)
                   
                    all_questions.extend(filtered_final_questions)
                    questions = list(set(all_questions))
                else:
                    questions = all_questions
            else:
                questions = unique_questions

            # Filter questions to ensure they're within the 8-15 word range
            filtered_questions = [q for q in questions if 8 <= len(q.split()) <= 15]
           
            # Sort questions by length (shortest first) to prioritize concise questions
            filtered_questions.sort(key=lambda q: len(q.split()))
           
            processed_questions = []
            # Divide into willingness levels (first 25 low, middle 25 medium, last 25 high)
            questions_to_use = filtered_questions[:num_questions]
            thirds = len(questions_to_use) // 3
           
            for i, question in enumerate(questions_to_use):
                if i < thirds:
                    willingness = "low_willingness"
                elif i < 2 * thirds:
                    willingness = "medium_willingness"
                else:
                    willingness = "high_willingness"
                           
                processed_questions.append({
                    "question": question,
                    "category_type": category_type,
                    "category": category,
                    "willingness_level": willingness
                })

            # Save questions to file
            if processed_questions:
                file_path = self._get_category_file_path(category_type, category)
                with open(file_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=["question", "category_type", "category", "willingness_level"])
                    writer.writeheader()
                    writer.writerows(processed_questions)

            return processed_questions

        except Exception as e:
            print(f"Error generating questions for {category_type}: {category}: {e}")
            return []

    def _get_category_file_path(self, category_type: str, category: str) -> Path:
        """Get the file path for a specific category's questions."""
        sanitized_category = category.lower().replace(" ", "_")
        return self.questions_dir / f"{category_type}_{sanitized_category}_questions.csv"

def generate_dataset_background(question_generator, category_type, category_name, system, ready_event=None):
    """Generate a dataset in the background and update the system when done."""
    try:
        print(f"\nBackground: Generating questions for {category_type}: {category_name}...")
        questions = question_generator._generate_category_questions(
            category_type=category_type,
            category=category_name,
            num_questions=75  # 25 questions per willingness level
        )
       
        # Update the system's all_questions cache to include the new questions
        if questions:
            with threading.Lock():
                for q in questions:
                    system.all_questions.add(q["question"])
           
            # Mark this dataset as available in the system
            system.update_available_datasets(category_type)
           
            print(f"\nBackground: Successfully generated dataset for {category_type}: {category_name}. Now available for interview.")
           
            # Signal that the dataset is ready
            if ready_event:
                ready_event.set()
        else:
            print(f"\nBackground: Failed to generate questions for {category_type}: {category_name}. Using default questions.")
    except Exception as e:
        print(f"\nBackground: Error generating questions for {category_type}: {category_name}: {str(e)}")
        # Make sure we still set the event even if there's an error
        if ready_event:
            ready_event.set()
