class FollowUpGenerator:
    def __init__(self, client):
        self.client = client
        self.context_history = deque(maxlen=3)  # Keep track of recent context
        self.last_followup = None
        
    def generate_follow_up(self, candidate_response, keywords, word_count, is_short, themes, missing_star):
        # Only generate follow-up if response is substantial
        if not candidate_response or len(candidate_response.split()) < 15:
            return None
            
        # Update context history
        self.context_history.append(candidate_response)
        
        # Prepare context for the LLM
        recent_context = "\n".join([f"- {resp}" for resp in self.context_history])
        
        prompt = f"""Generate one 8-12 word follow-up question based on the conversation context below.
The question should be natural, relevant, and help explore the candidate's personality and experiences more deeply.

Recent Conversation Context:
{recent_context}

Guidelines:
1. Focus on the most recent response but consider the full context
2. Ask about specific details mentioned (people, projects, challenges)
3. Explore motivations, feelings, or lessons learned
4. Connect to broader themes when appropriate
5. Keep it conversational and natural
6. Avoid yes/no questions
7. Make it personal but professional

Examples of good follow-ups:
- "What was the most surprising part of that experience for you?"
- "How did that situation change your approach to similar challenges?"
- "What would you do differently if you faced that again?"
- "Who influenced you most during that time?"

Generate just one follow-up question (8-12 words):"""
        
        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract and clean the question
            question = response.content[0].text.strip()
            
            # Remove any quotation marks and numbering
            question = re.sub(r'^[\d."\']+\s*', '', question).strip()
            
            # Validate the question
            if not self._is_valid_question(question):
                return None
                
            # Avoid repeating the same follow-up
            if question == self.last_followup:
                return None
                
            self.last_followup = question
            return question
            
        except Exception as e:
            print(f"Error generating follow-up: {e}")
            return None
            
    def _is_valid_question(self, question):
        """Validate the generated question meets our criteria"""
        if not question or not question.endswith('?'):
            return False
            
        words = question.split()
        if len(words) < 8 or len(words) > 12:
            return False
            
        # Check for common issues
        blacklist = [
            "can you", "could you", "would you", 
            "do you", "did you", "have you",
            "is there", "are there", "will you"
        ]
        
        if any(q.lower().startswith(tuple(blacklist)) for q in [question, question.split('?')[0]]):
            return False
            
        return True


def extract_keywords(text):
    """Extract important keywords from text"""
    if not text:
        return []
    
    keywords = ["team", "lead", "pressure", "conflict", "challenge", "problem", 
               "solution", "manage", "achieve", "result", "learn", "change"]
    
    tokens = word_tokenize(text.lower())
    pos_tags = pos_tag(tokens)
    
    found = [word for word in keywords if word in text.lower()]
    found += [word for word, pos in pos_tags if pos.startswith('NN') and len(word) > 3]
    found += [word for word, pos in pos_tags if pos.startswith('VB') and len(word) > 3]
    
    return list(set(found))[:5]

def identify_themes(text):
    """Identify psychological themes in response"""
    if not text:
        return []
    
    themes = []
    text_lower = text.lower()
    
    theme_keywords = {
        "teamwork": ["team", "collaborat", "work with"],
        "leadership": ["lead", "manage", "direct"],
        "stress": ["stress", "pressure", "tense"],
        "conflict": ["conflict", "disagree", "argument"],
        "achievement": ["achieve", "success", "result"],
        "learning": ["learn", "grow", "develop"]
    }
    
    for theme, keywords in theme_keywords.items():
        if any(kw in text_lower for kw in keywords):
            themes.append(theme)
    
    return themes if themes else ["experience"]

def check_star(text):
    """Check for missing STAR (Situation-Task-Action-Result) components"""
    missing = []
    
    if not re.search(r'(when|while|during|situation|circumstance)', text, re.I):
        missing.append("Situation")
    
    if not re.search(r'(task|goal|objective|needed to|had to)', text, re.I):
        missing.append("Task")
    
    if not re.search(r'(did|action|step|implement|decided)', text, re.I):
        missing.append("Action")
    
    if not re.search(r'(result|outcome|achieved|accomplished|learned)', text, re.I):
        missing.append("Result")
    
    return missing
