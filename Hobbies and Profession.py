def extract_hobbies(response_text):
    """
    Extract hobbies using NLP techniques for more flexible matching, ensuring verbs like 'play' and 'watch' have associated objects
    """
    # Handle empty response
    if not response_text:
        return []
       
    # Download required NLTK data (only needed once)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
       
    lemmatizer = WordNetLemmatizer()
   
    # Common hobby indicators and verbs
    hobby_indicators = ["enjoy", "like", "love", "passion", "hobby", "interest", "into", "fan of"]
   
    # Verbs that require objects to be considered hobbies
    object_requiring_verbs = {"play", "playing", "watch", "watching"}
   
    # Standalone common hobbies that don't need objects
    standalone_hobbies = {
        "dancing", "singing", "painting", "drawing", "reading", "writing", "cooking", "baking",
        "hiking", "swimming", "running", "cycling", "gaming", "gardening", "programming",
        "coding", "photography", "traveling", "collecting"
    }
   
    # Normalize text
    response_text = response_text.lower()
   
    # Tokenize and tag parts of speech
    tokens = word_tokenize(response_text)
    tagged = pos_tag(tokens)
   
    potential_hobbies = []
   
    # Step 1: Look for hobby indicators followed by verb phrases or noun phrases
    for i, (word, tag) in enumerate(tagged):
        if word in hobby_indicators and i < len(tagged) - 1:
            # Look ahead for potential multi-word hobby phrases
            j = i + 1
            phrase_words = []
            verb_found = None
           
            # Collect words that could be part of a hobby phrase
            while j < len(tagged) and tagged[j][1] not in ['.', ',', 'CC'] and j < i + 6:
                current_word = tagged[j][0]
                phrase_words.append(current_word)
               
                # Check if we found a verb that requires an object
                if current_word in object_requiring_verbs:
                    verb_found = current_word
                j += 1
           
            if phrase_words:
                # Form potential hobby phrase
                potential_phrase = ' '.join(phrase_words)
               
                # If phrase contains a verb requiring an object, ensure it has one
                if verb_found and len(phrase_words) > 1:
                    potential_hobbies.append(potential_phrase)
                # Otherwise add if it doesn't contain object-requiring verbs
                elif not any(word in object_requiring_verbs for word in phrase_words):
                    potential_hobbies.append(potential_phrase)
                # Special case: If the phrase is just a hobby-requiring verb, don't add it
                elif len(phrase_words) == 1 and phrase_words[0] in object_requiring_verbs:
                    continue
   
    # Step 2: Look for standalone hobbies directly
    for word, tag in tagged:
        if word in standalone_hobbies:
            potential_hobbies.append(word)
   
    # Step 3: Look for object-requiring verbs followed by their objects
    for i, (word, tag) in enumerate(tagged):
        if word in object_requiring_verbs and i < len(tagged) - 1:
            j = i + 1
            objects = []
           
            # Skip articles, determiners, adjectives
            while j < len(tagged) and tagged[j][1] in ['DT', 'JJ', 'RB']:
                j += 1
           
            # Collect the object(s)
            while j < len(tagged) and tagged[j][1] in ['NN', 'NNS', 'NNP', 'NNPS'] and j < i + 4:
                objects.append(tagged[j][0])
                j += 1
           
            if objects:
                # Form the hobby phrase with verb + objects
                hobby_phrase = word + " " + " ".join(objects)
                potential_hobbies.append(hobby_phrase)
   
    # Step 4: Look for domain-specific hobby phrases
    hobby_phrases = {
        "watching tv", "watching movies", "playing games", "playing video games",
        "playing music", "listening to music", "working out", "martial arts",
        "playing chess", "playing guitar", "playing piano", "watching sports",
        "playing football", "playing cricket", "watching football", "watching cricket"
    }
   
    # Create n-grams from the tokens to match phrases
    for n in range(2, 5):  # Check 2-word to 4-word phrases
        ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        for phrase in ngrams:
            if phrase in hobby_phrases:
                potential_hobbies.append(phrase)
   
    # Remove duplicates and generalize
    unique_hobbies = set()
    for hobby in potential_hobbies:
        # Filter out standalone object-requiring verbs
        if hobby in object_requiring_verbs:
            continue
       
        # Standardize common variations
        if hobby in ["watch tv", "watching television", "watch television"]:
            hobby = "watching TV"
        elif hobby in ["play video games", "playing video games", "video games", "video game"]:
            hobby = "playing video games"
        elif hobby in ["play games", "playing games"]:
            hobby = "playing games"
       
        # Only add if standalone hobby or if object-requiring verb has an object
        words = hobby.split()
        if (
            (len(words) == 1 and words[0] not in object_requiring_verbs) or
            (words[0] in object_requiring_verbs and len(words) > 1) or
            (words[0] not in object_requiring_verbs)
        ):
            unique_hobbies.add(hobby)
   
    # Return list instead of set for predictable order
    return list(unique_hobbies)

class ProfessionRecognizer:
    """
    A comprehensive system for recognizing and categorizing professional roles
    using natural language processing techniques.
    """
   
    def __init__(self):
        """Initialize the profession recognizer with industry definitions and NLP tools"""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
       
        # Enhanced industry definitions with weighted terms
        self.industry_definitions = self._initialize_industry_definitions()
       
        # Job levels (seniority)
        self.job_levels = {
            "entry": ["junior", "trainee", "entry", "associate", "assistant", "apprentice", "intern"],
            "mid": ["", "regular", "standard", "professional"],
            "senior": ["senior", "lead", "principal", "head", "chief", "experienced", "expert", "manager", "supervisor"],
            "executive": ["director", "executive", "vp", "vice president", "cxo", "officer",
                         "president", "founder", "chairman", "chairperson", "ceo", "cto", "cfo", "cio"]
        }
        
        # Academic positions
        self.academic_positions = ["principal", "headmaster", "headmistress", "head teacher"]
       
        # Common departments across industries
        self.departments = [
            "sales", "marketing", "finance", "accounting", "human resources", "hr", "operations",
            "it", "information technology", "research", "development", "research and development",
            "r&d", "legal", "administration", "customer service", "support", "logistics",
            "manufacturing", "production", "quality assurance", "qa", "engineering", "design",
            "product", "strategy", "business development", "communications", "public relations", "pr",
            "artificial intelligence", "ai", "machine learning", "ml", "data science"
        ]
       
        # Additional specialized departments by industry
        self.specialized_departments = {
            "Education": ["academics", "admissions", "faculty", "student affairs", "curriculum",
                         "teaching", "instruction", "examination", "assessment", "learning"],
            "Healthcare": ["nursing", "pharmacy", "emergency", "radiology", "cardiology", "neurology",
                          "pediatrics", "obstetrics", "gynecology", "oncology", "surgery", "anesthesia"],
            "Technology": ["software", "hardware", "networking", "security", "data", "ai",
                          "machine learning", "cloud", "infrastructure", "devops", "web", "mobile",
                          "data science", "artificial intelligence", "computer vision", "natural language processing",
                          "robotics", "deep learning", "neural networks", "big data", "augmented reality",
                          "virtual reality", "blockchain", "quantum computing", "cybersecurity",
                          "mechanical engineering", "civil engineering"],  # Added mechanical and civil engineering
            "Science": ["laboratory", "experimental", "theoretical", "computational", "environmental",
                       "biological", "chemical", "physical", "geological", "astronomical"]
        }
       
        # Initialize spaCy NER patterns
        if nlp:
            self._initialize_spacy_patterns()
   
    def _initialize_industry_definitions(self):
        """Define comprehensive industry categories with weighted terms"""
        return {
            "Technology": {
                "core_roles": ["developer", "engineer", "programmer", "architect", "administrator",
                             "scientist", "analyst", "specialist", "consultant", "technician", "officer",
                             "expert", "researcher", "strategist", "lead", "designer", "technical", "professional"],
                "core_fields": ["software", "hardware", "data", "network", "system", "cloud", "web",
                              "mobile", "app", "security", "infrastructure", "IT", "tech", "digital",
                              "artificial intelligence", "machine learning", "ai", "ml", "deep learning",
                              "computer vision", "nlp", "natural language processing", "big data",
                              "computer science", "data science", "computation", "analytics",
                              "mechanical engineering", "civil engineering"],  # Added here
                "associated_terms": ["code", "programming", "development", "agile", "scrum", "devops",
                                   "full-stack", "front-end", "back-end", "database", "server", "API",
                                   "cybersecurity", "artificial intelligence", "machine learning", "AI",
                                   "ML", "deep learning", "neural network", "blockchain", "IoT", "UX", "UI",
                                   "algorithm", "python", "java", "javascript", "cloud", "aws", "azure",
                                   "data mining", "predictive modeling", "statistical analysis",
                                   "tensorflow", "pytorch", "computer vision", "nlp", "transformers",
                                   "automation", "robotics", "digital transformation",
                                   "mechanical", "civil"],  # Added here
                "specific_roles": ["software engineer", "data scientist", "web developer", "system administrator",
                                 "network engineer", "security analyst", "cloud architect", "devops engineer",
                                 "UX designer", "product manager", "database administrator", "CTO",
                                 "AI engineer", "machine learning engineer", "data engineer", "AI researcher",
                                 "AI expert", "ML specialist", "artificial intelligence specialist",
                                 "NLP engineer", "computer vision engineer", "deep learning specialist",
                                 "research scientist", "AI scientist", "data architect", "AI strategist",
                                 "data analyst", "business intelligence analyst",
                                 "mechanical engineer", "civil engineer"]  # Added here
            },
            "Education": {
                "core_roles": ["professor", "teacher", "lecturer", "instructor", "educator", "tutor",
                             "researcher", "scholar", "academic", "faculty", "dean", "principal",
                             "administrator", "counselor", "coordinator"],
                "core_fields": ["education", "teaching", "academic", "learning", "pedagogy", "curriculum",
                              "instruction", "school", "university", "college", "faculty", "classroom"],
                "associated_terms": ["student", "class", "course", "lecture", "lesson", "syllabus",
                                   "assessment", "evaluation", "grade", "degree", "diploma", "thesis",
                                   "dissertation", "research", "educational", "scholarship", "training"],
                "specific_roles": ["department chair", "dean", "provost", "curriculum developer",
                                 "education consultant", "academic advisor", "research professor",
                                 "HOD", "head of department", "assistant professor", "associate professor",
                                 "principal"]  # Added principal here
            },
            "Healthcare": {
                "core_roles": ["doctor", "physician", "surgeon", "nurse", "practitioner", "therapist",
                             "specialist", "technician", "assistant", "aide", "counselor", "clinician"],
                "core_fields": ["health", "medicine", "medical", "clinical", "therapy", "nursing",
                              "patient", "care", "treatment", "diagnosis", "hospital", "clinic"],
                "associated_terms": ["healthcare", "wellness", "healing", "doctor", "nurse", "practitioner",
                                   "therapeutic", "preventive", "rehabilitation", "disease", "illness",
                                   "syndrome", "prescription", "medication", "surgery", "emergency"],
                "specific_roles": ["cardiologist", "neurologist", "registered nurse", "physical therapist",
                                 "medical technician", "psychiatrist", "pediatrician", "surgeon",
                                 "radiologist", "pharmacist", "anesthesiologist", "dentist"]
            },
            "Business": {
                "core_roles": ["manager", "executive", "director", "analyst", "consultant", "specialist",
                             "coordinator", "representative", "agent", "advisor", "officer", "associate"],
                "core_fields": ["business", "management", "marketing", "sales", "finance", "accounting",
                              "operations", "administration", "hr", "strategy", "consulting"],
                "associated_terms": ["corporate", "commercial", "enterprise", "company", "firm", "client",
                                   "market", "revenue", "profit", "growth", "startup", "leadership",
                                   "organization", "project", "process", "budget", "forecast", "reporting"],
                "specific_roles": ["marketing manager", "financial analyst", "HR director", "operations manager",
                                 "business consultant", "project manager", "sales representative",
                                 "account executive", "CEO", "CFO", "COO"]
            },
            "Legal": {
                "core_roles": ["lawyer", "attorney", "counsel", "judge", "clerk", "paralegal",
                             "advocate", "mediator", "arbitrator", "solicitor", "barrister"],
                "core_fields": ["law", "legal", "justice", "court", "litigation", "compliance",
                              "regulatory", "legislation", "judicial", "contract"],
                "associated_terms": ["lawsuit", "case", "trial", "settlement", "ruling", "judgment",
                                   "prosecution", "defense", "plaintiff", "defendant", "client",
                                   "statute", "regulation", "rights", "intellectual property", "IP",
                                   "patent", "trademark", "copyright", "corporate law", "criminal law"],
                "specific_roles": ["corporate lawyer", "patent attorney", "trial lawyer", "judge",
                                 "general counsel", "legal advisor", "compliance officer",
                                 "legal consultant", "legal director", "public defender"]
            },
            "Creative": {
                "core_roles": ["designer", "artist", "writer", "creator", "director", "producer",
                             "editor", "developer", "consultant", "specialist", "coordinator"],
                "core_fields": ["design", "art", "content", "media", "creative", "visual",
                              "digital", "production", "entertainment", "communication"],
                "associated_terms": ["graphic", "web", "UX", "UI", "video", "audio", "animation",
                                   "illustration", "photography", "fashion", "interior", "broadcast",
                                   "publishing", "advertising", "marketing", "branding", "campaign",
                                   "creative", "artistic", "aesthetic", "composition", "layout"],
                "specific_roles": ["graphic designer", "content writer", "art director", "creative director",
                                 "UI/UX designer", "video producer", "fashion designer", "web designer",
                                 "interior designer", "multimedia artist", "animator"]
            },
            "Science": {
                "core_roles": ["scientist", "researcher", "analyst", "technician", "specialist",
                             "fellow", "assistant", "associate", "professor", "engineer"],
                "core_fields": ["research", "science", "laboratory", "experimental", "computational",
                              "theoretical", "environmental", "biological", "chemical", "physical"],
                "associated_terms": ["experiment", "analysis", "study", "investigation", "discovery",
                                   "innovation", "publication", "journal", "data", "observation",
                                   "hypothesis", "theory", "methodology", "protocol", "grant", "funding"],
                "specific_roles": ["research scientist", "lab technician", "principal investigator",
                                 "biologist", "chemist", "physicist", "geologist", "astronomer",
                                 "ecologist", "meteorologist", "neuroscientist"]
            },
            "Manufacturing": {
                "core_roles": ["engineer", "technician", "operator", "manager", "supervisor",
                             "specialist", "designer", "coordinator", "planner", "inspector"],
                "core_fields": ["manufacturing", "production", "assembly", "industrial", "fabrication",
                              "processing", "quality", "operations", "plant", "factory"],
                "associated_terms": ["equipment", "machinery", "automation", "robotic", "lean", "six sigma",
                                   "inventory", "supply chain", "logistics", "maintenance", "safety",
                                   "inspection", "quality control", "QA", "QC", "CAD", "CAM"],
                "specific_roles": ["production manager", "quality engineer", "process engineer",
                                 "manufacturing technician", "industrial designer", "plant manager",
                                 "supply chain specialist", "production planner", "machinist"]
            },
            "Finance": {
                "core_roles": ["analyst", "manager", "advisor", "consultant", "specialist",
                             "planner", "broker", "trader", "banker", "accountant", "auditor"],
                "core_fields": ["finance", "financial", "investment", "banking", "accounting",
                              "wealth", "asset", "risk", "tax", "audit", "treasury"],
                "associated_terms": ["money", "capital", "fund", "portfolio", "stock", "bond", "security",
                                   "market", "trading", "investment", "loan", "credit", "debt",
                                   "mortgage", "insurance", "budget", "forecast", "analysis"],
                "specific_roles": ["financial analyst", "investment banker", "portfolio manager",
                                 "financial advisor", "accountant", "auditor", "tax specialist",
                                 "risk analyst", "loan officer", "CFO", "treasurer"]
            }
        }
   
    def _initialize_spacy_patterns(self):
        """Initialize spaCy patterns for better entity recognition"""
        # This would add custom entity patterns to improve recognition
        # Implementation depends on specific needs
        pass
   
    def process_profession(self, profession_text, years_experience=None):
        """
        Main entry point to process a profession description
       
        :param profession_text: Text describing the profession
        :param years_experience: Years of experience (optional)
        :return: Dictionary with profession categorization details
        """
        # Preprocess the input text
        processed_text = self._preprocess_text(profession_text)
       
        # Extract structural components
        components = self._extract_components(profession_text)
       
        # If spaCy is available, enhance entity recognition
        if nlp:
            spacy_entities = self._extract_spacy_entities(profession_text)
            components.update(spacy_entities)
       
        # Determine industry categories with scores
        category_scores = self._score_categories(processed_text, components)
       
        # Check for interdisciplinary characteristics
        interdisciplinary = self._check_interdisciplinary(category_scores)
       
        # Generate appropriate job title/subcategory
        if interdisciplinary:
            primary_cat, secondary_cat = interdisciplinary
            subcategory = self._generate_interdisciplinary_title(
                components, processed_text, primary_cat, secondary_cat, years_experience
            )
            result = {
                "main_category": primary_cat,
                "subcategory": subcategory,
                "secondary_category": secondary_cat,
                "roles": components.get("roles", []),
                "departments": components.get("departments", []),
                "interdisciplinary": True,
                "category_scores": category_scores  # Added for transparency
            }
        else:
            # Single primary category
            if category_scores:
                main_category = max(category_scores, key=category_scores.get)
                subcategory = self._generate_title(
                    components, processed_text, main_category, years_experience
                )
                result = {
                    "main_category": main_category,
                    "subcategory": subcategory,
                    "roles": components.get("roles", []),
                    "departments": components.get("departments", []),
                    "interdisciplinary": False,
                    "category_scores": category_scores  # Added for transparency
                }
            else:
                # Fallback for unrecognized professions
                result = self._handle_unrecognized_profession(profession_text, years_experience)
       
        # Add seniority level if available
        if years_experience is not None:
            result["seniority"] = self._determine_seniority(years_experience, components)
       
        return result
   
    def _preprocess_text(self, text):
        """Clean and normalize text for processing"""
        if not text:
            return ""
           
        # Convert to lowercase
        text = text.lower()
       
        # Remove punctuation except hyphens and apostrophes
        text = re.sub(r'[^\w\s\'-]', ' ', text)
       
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
       
        return text
   
    def _extract_components(self, text):
        """Extract structural components from profession text"""
        components = {
            "roles": [],
            "departments": [],
            "organizations": [],
            "fields": [],
            "levels": [],
            "technologies": []  # Added for tech stack identification
        }
       
        # Preprocess text for better extraction
        processed_text = self._preprocess_text(text)
       
        # Extract organization types
        org_types = ["company", "university", "college", "school", "hospital",
                    "firm", "agency", "corporation", "institute", "laboratory", "startup"]
       
        for org in org_types:
            if org in processed_text:
                components["organizations"].append(org)
       
        # Extract departments
        all_departments = self.departments.copy()
        for industry, deps in self.specialized_departments.items():
            all_departments.extend(deps)
           
        for dept in all_departments:
            if dept in processed_text:
                components["departments"].append(dept)
               
        # Look for "X department" pattern
        dept_patterns = [
            r"(\w+(?:\s+\w+)?)\s+department",
            r"department\s+of\s+(\w+(?:\s+\w+)?)",
            r"(\w+(?:\s+\w+)?)\s+division",
            r"(\w+(?:\s+\w+)?)\s+faculty"
        ]
       
        for pattern in dept_patterns:
            matches = re.findall(pattern, processed_text)
            if matches:
                for match in matches:
                    if match not in components["departments"]:
                        components["departments"].append(match)
       
        # Extract technologies
        tech_keywords = [
            "python", "java", "javascript", "c++", "ruby", "php", "swift", "kotlin", "go", "rust",
            "react", "angular", "vue", "node", "django", "flask", "spring", "tensorflow", "pytorch",
            "pandas", "numpy", "scikit-learn", "sql", "nosql", "mongodb", "postgresql", "mysql",
            "aws", "azure", "gcp", "cloud", "kubernetes", "docker", "devops", "ci/cd", "git"
        ]
       
        for tech in tech_keywords:
            if tech in processed_text:
                components["technologies"].append(tech)
       
        # Extract AI-specific terms
        ai_terms = [
            "machine learning", "deep learning", "neural network", "nlp", "computer vision",
            "natural language processing", "reinforcement learning", "supervised learning",
            "unsupervised learning", "data mining", "big data", "artificial intelligence", "ai",
            "ml", "chatbot", "recommendation system", "predictive model", "clustering", "classification"
        ]
       
        for term in ai_terms:
            if term in processed_text:
                components["fields"].append(term)
                # For AI terms, also add them to technologies
                if term not in components["technologies"]:
                    components["technologies"].append(term)
       
        # Extract roles using NLTK
        tokens = word_tokenize(processed_text)
        pos_tags = nltk.pos_tag(tokens)
        
        # Special handling for "principal" - check academic context
        if "principal" in processed_text:
            academic_context = any(word in processed_text for word in 
                                 ["school", "college", "academy", "education", "teacher", "student"])
            if academic_context:
                components["roles"].append("principal")
                components["levels"].append("executive")  # Principal is a high-level position in education
            else:
                # It's likely the seniority meaning
                components["levels"].append("principal")
       
        # First pass - look for key role indicators
        key_professional_terms = ["expert", "specialist", "professional", "engineer", "scientist",
                                 "developer", "analyst", "consultant", "advisor", "manager"]
       
        for term in key_professional_terms:
            if term in processed_text:
                components["roles"].append(term)
       
        # Second pass - extract potential roles based on part-of-speech
        for i, (word, tag) in enumerate(pos_tags):
            # Look for nouns that could be roles
            if tag.startswith('NN'):
                # Check if it's in any of our core roles across industries
                for industry, definition in self.industry_definitions.items():
                    if word in definition["core_roles"]:
                        if word not in components["roles"]:
                            components["roles"].append(word)
                        break
       
        # Extract specific roles from definitions - now with phrase detection
        for industry, definition in self.industry_definitions.items():
            for role in definition["specific_roles"]:
                if role in processed_text:
                    if role not in components["roles"]:
                        components["roles"].append(role)
                       
                # Also look for bigrams and trigrams for better phrase detection
                if len(role.split()) > 1:
                    role_words = role.split()
                    # Check if all words appear close together
                    if all(word in processed_text for word in role_words):
                        proximity_score = 0
                        for i in range(len(tokens)-1):
                            if tokens[i] in role_words and tokens[i+1] in role_words:
                                proximity_score += 1
                       
                        # If words appear in proximity, likely a phrase
                        if proximity_score > 0 and role not in components["roles"]:
                            components["roles"].append(role)
       
        # Extract job levels/seniority (excluding principal if already handled)
        for level, terms in self.job_levels.items():
            for term in terms:
                if term and term in processed_text and term != "principal":  # Skip principal if already handled
                    components["levels"].append(term)
       
        # Identify fields of expertise
        for industry, definition in self.industry_definitions.items():
            for field in definition["core_fields"]:
                if field in processed_text and field not in components["fields"]:
                    components["fields"].append(field)
       
        # Special handling for AI/Tech roles
        if "artificial intelligence" in processed_text or "ai" in processed_text.split():
            if "ai" not in components["fields"]:
                components["fields"].append("ai")
            if "artificial intelligence" not in components["fields"]:
                components["fields"].append("artificial intelligence")
               
        if "data science" in processed_text:
            if "data science" not in components["fields"]:
                components["fields"].append("data science")
               
        # Hard-code common modern tech roles
        specific_tech_roles = {
            "ai expert": ["ai", "expert"],
            "ai engineer": ["ai", "engineer"],
            "artificial intelligence expert": ["artificial intelligence", "expert"],
            "machine learning engineer": ["machine learning", "engineer"],
            "data scientist": ["data", "scientist"],
            "data engineer": ["data", "engineer"],
            "data analyst": ["data", "analyst"],
            "deep learning engineer": ["deep learning", "engineer"],
            "nlp engineer": ["nlp", "engineer", "natural language processing"],
            "cloud architect": ["cloud", "architect"],
            "devops engineer": ["devops", "engineer"],
            "mechanical engineer": ["mechanical", "engineer"],  # Added
            "civil engineer": ["civil", "engineer"]  # Added
        }
       
        # Check for these roles specifically
        for role_name, keywords in specific_tech_roles.items():
            if all(keyword in processed_text for keyword in keywords):
                if role_name not in components["roles"]:
                    components["roles"].append(role_name)
       
        return components
   
    def _extract_spacy_entities(self, text):
        """Use spaCy for enhanced entity recognition"""
        additional_components = defaultdict(list)
       
        doc = nlp(text)
       
        # Extract organizations
        for ent in doc.ents:
            if ent.label_ == "ORG":
                additional_components["organizations"].append(ent.text.lower())
       
        # Extract job titles (requires custom training or pattern matching)
        # This is simplified - would need more sophisticated patterns in practice
        job_patterns = [tok.text.lower() for tok in doc if tok.pos_ == "NOUN"]
        for pattern in job_patterns:
            for industry in self.industry_definitions:
                if pattern in self.industry_definitions[industry]["core_roles"]:
                    additional_components["roles"].append(pattern)
       
        return dict(additional_components)
   
    def _score_categories(self, text, components):
        """Score each industry category based on extracted components"""
        scores = defaultdict(int)
       
        # Process each industry definition
        for industry, definition in self.industry_definitions.items():
            # Score based on core roles (high weight)
            for role in definition["core_roles"]:
                if role in text:
                    scores[industry] += 3
                # Check against extracted roles
                if role in components.get("roles", []):
                    scores[industry] += 3
           
            # Score based on specific roles (highest weight)
            for role in definition["specific_roles"]:
                if role in text:
                    scores[industry] += 5
                if role in components.get("roles", []):
                    scores[industry] += 5
           
            # Score based on core fields (medium weight)
            for field in definition["core_fields"]:
                if field in text:
                    scores[industry] += 2
                if field in components.get("fields", []):
                    scores[industry] += 2
           
            # Score based on associated terms (low weight)
            for term in definition["associated_terms"]:
                if term in text:
                    scores[industry] += 1
       
        # Special adjustments for certain industries
       
        if "professor" in text:
            scores["Education"] += 10  # Heavy boost to ensure Education wins
            # Reduce Technology score if it's an engineering professor
            if any(field in text for field in ["engineering", "mechanical", "civil"]):
                scores["Technology"] = max(0, scores["Technology"] - 2)

        # Boost technology score for AI/data related terms
        ai_terms = ["ai", "artificial intelligence", "machine learning", "data science",
                   "deep learning", "neural network", "nlp", "computer vision"]
        for term in ai_terms:
            if term in text or term in components.get("fields", []):
                scores["Technology"] += 3  # Significant boost
       
        # Special case: adjust scores for academic positions in technical fields
        if any(role in text for role in ["professor", "lecturer", "instructor"]) and \
           any(dept in text for dept in ["computer science", "engineering", "technology", "ai", "data science"]):
            # This is likely an academic in a technical field - boost both categories
            scores["Education"] += 2
            scores["Technology"] += 1
            
        # Special case: mechanical/civil engineering
        if "mechanical engineer" in text or "mechanical engineering" in text:
            scores["Technology"] += 4
        if "civil engineer" in text or "civil engineering" in text:
            scores["Technology"] += 4
       
        # Significant boost for explicit mentions of industry
        industry_indicators = {
            "Technology": ["tech company", "tech industry", "technology sector", "tech field",
                          "software company", "it company", "startup"],
            "Healthcare": ["hospital", "clinic", "healthcare provider", "medical center"],
            "Education": ["university", "college", "school", "academy", "educational institution"],
            "Finance": ["bank", "financial institution", "investment firm", "hedge fund"],
            "Legal": ["law firm", "legal practice", "attorney office"],
            "Creative": ["agency", "studio", "design firm", "creative company"],
            "Manufacturing": ["factory", "plant", "manufacturing facility"],
            "Science": ["laboratory", "research institution", "scientific facility"]
        }
       
        for industry, indicators in industry_indicators.items():
            for indicator in indicators:
                if indicator in text:
                    scores[industry] += 4  # Strong indicator
       
        # Special handling for AI expertise
        if "artificial intelligence expert" in text or "ai expert" in text:
            scores["Technology"] += 8  # Very strong signal
           
        # Remove Creative category score for clear technology roles
        # (This fixes the AI expert misclassification)
        clear_tech_indicators = ["artificial intelligence", "machine learning", "data science",
                                "software development", "web development", "cybersecurity",
                                "mechanical engineering", "civil engineering"]  # Added
        if any(indicator in text for indicator in clear_tech_indicators):
            tech_related = True
            creative_related = False
           
            # Only if there are genuine creative terms
            creative_indicators = ["artist", "illustration", "painting", "drawing", "composer",
                                  "musician", "writer", "author", "filmmaker"]
           
            if any(indicator in text for indicator in creative_indicators):
                creative_related = True
               
            # If tech related but not genuinely creative, reduce creative score
            if tech_related and not creative_related:
                scores["Creative"] = max(0, scores["Creative"] - 3)
       
        return dict(scores)
   
    def _check_interdisciplinary(self, category_scores):
        """
        Check if the profession spans multiple disciplines
        Returns tuple of (primary_category, secondary_category) or None
        """
        if not category_scores:
            return None
           
        # Get top categories by score
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
       
        # We need at least two categories with meaningful scores
        if len(sorted_categories) >= 2 and sorted_categories[0][1] > 0 and sorted_categories[1][1] > 0:
            # Check if the secondary category has a significant score
            # (at least 70% of the primary category's score)
            primary_score = sorted_categories[0][1]
            secondary_score = sorted_categories[1][1]
           
            if secondary_score >= primary_score * 0.7:
                return (sorted_categories[0][0], sorted_categories[1][0])
       
        return None
   
    def _generate_title(self, components, text, category, years_experience=None):
        """Generate an appropriate job title/subcategory"""
        # Try to build a title from extracted components
        roles = components.get("roles", [])
        levels = components.get("levels", [])
        departments = components.get("departments", [])
        fields = components.get("fields", [])
        technologies = components.get("technologies", [])
        
        # Special handling for academic principals
        if "principal" in roles and any(word in text for word in ["school", "college", "academy"]):
            org_type = next((word for word in ["school", "college", "academy"] if word in text), "school")
            return f"Principal of {org_type.title()}"

        # First check for AI/Tech specific roles
        if category == "Technology":
            # Check for AI expert specifically
            if "artificial intelligence expert" in roles:
                return "Artificial Intelligence Expert"
            if "ai expert" in roles:
                return "AI Expert"
            if "expert" in roles and ("ai" in fields or "artificial intelligence" in fields):
                return "AI Expert"
               
            # Check for data scientist
            if "data scientist" in roles:
                return "Data Scientist"
            if "scientist" in roles and "data" in fields:
                return "Data Scientist"
               
            # Check for ML/AI engineers
            if "machine learning engineer" in roles:
                return "Machine Learning Engineer"
            if "ai engineer" in roles or "artificial intelligence engineer" in roles:
                return "AI Engineer"
            if "engineer" in roles:
                if "machine learning" in fields:
                    return "Machine Learning Engineer"
                if "ai" in fields or "artificial intelligence" in fields:
                    return "AI Engineer"
                if "mechanical" in fields:
                    return "Mechanical Engineer"
                if "civil" in fields:
                    return "Civil Engineer"
       
        # Start with seniority level if available
        title_parts = []
       
        # Add level if present
        if levels:
            level = max(levels, key=len).title()  # Use the longest level term
            title_parts.append(level)
        elif years_experience is not None:
            # Use experience to determine level if not explicitly stated
            experience_level = self._get_experience_level_term(years_experience)
            if experience_level and experience_level != "mid":  # Skip mid-level as it's often implied
                title_parts.append(experience_level.title())
       
        # Add the main role if available
        if roles:
            # Find the most specific role (usually the longest)
            main_role = max(roles, key=len).title()  # Use the longest role as it's likely most specific
            title_parts.append(main_role)
        else:
            # Fallback - look for nouns that might be roles
            tokens = word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            for word, tag in pos_tags:
                if tag.startswith('NN') and len(word) > 3:
                    title_parts.append(word.title())
                    break
       
        # Add department or field context if available
        if departments:
            dept = max(departments, key=len).title()  # Use longest/most specific
            # Avoid redundancy
            if not any(dept.lower() in part.lower() for part in title_parts):
                # Format appropriately
                if "department" in dept.lower():
                    title_parts.append(f"of {dept}")
                else:
                    title_parts.append(f"in {dept}")
        elif fields:
            field = max(fields, key=len).title()
            if not any(field.lower() in part.lower() for part in title_parts):
                title_parts.append(f"in {field}")
       
        # Construct the title
        if title_parts:
            return " ".join(title_parts)
       
        # Ultimate fallback
        return self._get_experience_level_term(years_experience).title() + " Professional"
   
    def _generate_interdisciplinary_title(self, components, text, primary_cat, secondary_cat, years_experience=None):
        """Generate title for interdisciplinary roles"""
        roles = components.get("roles", [])
        departments = components.get("departments", [])
        fields = components.get("fields", [])
       
        # Handle common interdisciplinary combinations
       
        # Tech + Science combinations (e.g., Research Scientist in AI)
        if (primary_cat == "Technology" and secondary_cat == "Science") or (primary_cat == "Science" and secondary_cat == "Technology"):
            # AI/ML Research roles
            if "researcher" in roles or "scientist" in roles:
                if any(field in fields for field in ["ai", "artificial intelligence", "machine learning", "deep learning"]):
                    ai_field = next((f for f in fields if f in ["artificial intelligence", "machine learning", "deep learning"]), "AI")
                    return f"Research Scientist in {ai_field.title()}"
       
        # Tech + Education combinations (e.g., Professor of Computer Science)
        if (primary_cat == "Education" and secondary_cat == "Technology") or (primary_cat == "Technology" and secondary_cat == "Education"):
            if any(role in roles for role in ["professor", "teacher", "lecturer"]):
                if any(dept in departments for dept in ["computer science", "data science", "ai"]):
                    tech_dept = next((d for d in departments if d in ["computer science", "data science", "ai"]), "")
                    return f"{tech_dept.upper() if tech_dept.lower() == 'ai' else tech_dept.title()} Professor"
       
        # Tech + Creative combinations (e.g., UX Designer)
        if (primary_cat == "Technology" and secondary_cat == "Creative") or (primary_cat == "Creative" and secondary_cat == "Technology"):
            if "designer" in roles:
                if any(field in fields for field in ["ux", "ui", "user experience", "user interface"]):
                    return "UX/UI Designer"
       
        # Tech + Business combinations (e.g., Data Analytics Manager)
        if (primary_cat == "Technology" and secondary_cat == "Business") or (primary_cat == "Business" and secondary_cat == "Technology"):
            if "manager" in roles or "director" in roles or "head" in roles:
                if any(field in fields for field in ["data", "analytics", "ai", "machine learning"]):
                    tech_field = next((f for f in fields if f in ["data", "analytics", "ai", "machine learning"]), "Technology")
                    return f"{tech_field.title()} {max(roles, key=len).title()}"
       
        # General interdisciplinary title generation
        title = self._generate_title(components, text, primary_cat, years_experience)
       
        # For other cases, note the interdisciplinary nature
        return title
   
    def _handle_unrecognized_profession(self, text, years_experience=None):
        """Handle cases where profession can't be clearly categorized"""
        # Extract any noun as a potential profession
        tokens = word_tokenize(text.lower())
        pos_tags = nltk.pos_tag(tokens)
       
        profession = "Professional"
        for word, tag in pos_tags:
            if tag.startswith('NN') and len(word) > 3 and word not in self.stop_words:
                profession = word.title()
                break
       
        # Check for AI/technology specific terms
        if any(term in text.lower() for term in ["ai", "artificial intelligence", "machine learning", "data science"]):
            # This is likely a tech role
            tech_term = next((term for term in ["artificial intelligence", "machine learning", "data science", "ai"]
                             if term in text.lower()), "Technology")
           
            role_term = next((term for term in ["expert", "specialist", "engineer", "scientist", "analyst"]
                             if term in text.lower()), "Professional")
           
            tech_title = f"{tech_term.title()} {role_term.title()}"
           
            return {
                "main_category": "Technology",
                "subcategory": tech_title,
                "roles": [role_term.lower()],
                "fields": [tech_term.lower()],
                "departments": [],
                "confidence": "medium"
            }
       
        # Try to find any category that might be relevant
        for category, definition in self.industry_definitions.items():
            for term_list in definition.values():
                for term in term_list:
                    if term in text.lower():
                        return {
                            "main_category": category,
                            "subcategory": self._get_experience_level_title(years_experience, profession),
                            "roles": [profession.lower()],
                            "departments": [],
                            "confidence": "low"
                        }
       
        # Complete fallback
        return {
            "main_category": "Other",
            "subcategory": self._get_experience_level_title(years_experience, profession),
            "roles": [profession.lower()],
            "departments": [],
            "confidence": "very low"
        }
   
    def _determine_seniority(self, years_experience, components):
        """Determine seniority level from years of experience and job components"""
        # First check if explicit seniority terms are present
        levels = components.get("levels", [])
       
        for level_name, terms in self.job_levels.items():
            if any(term in levels for term in terms):
                return level_name
       
        # If no explicit terms, use years of experience
        return self._get_experience_level_term(years_experience)
   
    def _get_experience_level_term(self, years):
        """Get experience level term based on years"""
        if years is None:
            return "mid"
           
        if years < 2:
            return "entry"
        elif years < 5:
            return "mid"
        elif years < 10:
            return "senior"
        else:
            return "executive"
   
    def _get_experience_level_title(self, years, profession):
        """Generate title with experience level"""
        level = self._get_experience_level_term(years)
       
        if level == "entry":
            return f"Junior {profession}"
        elif level == "mid":
            return profession
        elif level == "senior":
            return f"Senior {profession}"
        else:
            return f"Executive {profession}"


# Function to use the recognizer
def recognize_profession(profession_text, years_experience=None):
    """
    Recognize and categorize a profession using NLP techniques
   
    :param profession_text: Description of the profession
    :param years_experience: Years of experience in the profession (optional)
    :return: Dictionary with categorization details
    """
    recognizer = ProfessionRecognizer()
    return recognizer.process_profession(profession_text, years_experience)

def extract_number_from_text(text):
    """Extract a number from text, handling both digits and number words."""
    # Check for digits first
    digit_match = re.search(r'(\d+\.?\d*)', text)
    if digit_match:
        return float(digit_match.group(1))

    # Dictionary of number words to values
    number_words = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50
    }

    # Check for number words
    for word, value in number_words.items():
        if word in text.lower():
            return float(value)

    return None
