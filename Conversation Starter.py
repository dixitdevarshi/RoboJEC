import numpy as np
import random
import time
from pathlib import Path
import csv
from typing import List, Dict, Any, Optional
from enum import Enum
from collections import deque
from anthropic import Anthropic
import os
import re
import pyttsx3
import tempfile
import speech_recognition as sr
import threading
import math
from datetime import datetime
import pyaudio
import librosa
import threading
import queue
import json
from transformers import pipeline
import torch
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.chunk import ne_chunk
import spacy
from collections import defaultdict
import statistics

#nltk.download('maxent_ne_chunker_tab')



class WillingnessLevel(Enum):
    LOW = "low_willingness"
    MEDIUM = "medium_willingness"
    HIGH = "high_willingness"


class ConversationStarterGenerator: 
    def __init__(self):
        
        self.greetings = [
            "Hello, good to see you!",
            "Hi there, wonderful to meet you!",
            "Greetings, it's a pleasure!",
            "Hello, what a delight to meet you!"
        ]
        self.ai_introductions = [
            "I am robojec, an AI system designed to facilitate insightful discussions. Welcome {name}, it's an absolute pleasure to have you here.",
            "I'm robojec, an artificial intelligence created for meaningful conversations. Welcome {name}, I'm truly excited to have you with us today.",
            "My name is robojec, an AI interview system. Welcome {name}, It's wonderful to have you here."
        ]

        self.followups = [
            "I'm robojec, an artificial intelligence created for meaningful conversations.",
            "I'm robojec, here to have an interesting discussion with you.",
            "My name is robojec - I'm an AI designed for engaging conversations."
        ]
        self.welcome=[
           "Welcome {name}, It's an absolute pleasure to have you here.",
           "Welcome {name}, I'm truly excited to have you with us today.",
           "Welcome {name}, It's wonderful to have you here."
        ]
        self.welcome2=[
            "I'm looking forward to delving into your thoughts on both your career and your interests. Let's start our conversation.",
            "I can't wait to explore your professional insights and personal passions. Let's start our conversation.",
            "I'm eager to learn about your work experiences and the things that interest you beyond your profession. Let's start our conversation."
        ]
        self.name_questions = [
            "What's your name?",
            "May I know your name?",
            "What should I call you?",
            "What name do you go by?",
            "I'm sorry, I didn't catch your name?",
            "Mind sharing your name?",
            "Who do I have the pleasure of speaking with?",
            "Can you tell me your name?"
        ]

       
        self.profession_questions = [
            "{name}, what is your profession?",  
            "{name}, what do you do professionally?",  
            "{name}, what's your current profession?", 
            "{name}, what is your official job title?",  
            "{name}, what is your current occupation?"
        ]
       
        self.experience_questions = [
            "{name}, how many years of experience do you have?",
            "How long have you been in this field, {name}?",
            "What's your total work experience, {name}?",
            "Can you share how many years you've been working, {name}?",
            "{name}, how experienced are you in your profession?",
            "How long have you been doing this work, {name}?"
        ]
       

        self.hobby_questions = [
            "What are some of your hobbies or activities you enjoy, {name}?",
            "Tell me about your interests outside of work, {name}. What do you enjoy doing?",
            "{name}, what activities do you find most fulfilling in your free time?",
            "I'd love to know about your hobbies, {name}. What do you like to do?",
            "What are some activities or interests that you're passionate about, {name}?",
            "{name}, how do you like to spend your leisure time?",
            "Outside of your professional life, what kinds of activities interest you, {name}?"
        ]

    def extract_name(self, input_text: str) -> str:
        # Ensure required NLTK data is downloaded
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('taggers/maxent_treebank_pos_tagger')
            nltk.data.find('chunkers/maxent_ne_chunker')
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('punkt')
            nltk.download('maxent_treebank_pos_tagger')
            nltk.download('maxent_ne_chunker')
            nltk.download('words')

        # Mapping of titles/designations to their formal representation
        title_expansions = {
            "dr": "Doctor", "dr.": "Doctor", "doctor": "Doctor",
            "mr": "Mister", "mr.": "Mister", "mister": "Mister",
            "mrs": "Missus", "mrs.": "Missus",
            "miss": "Miss", "ms": "Miss", "ms.": "Miss",
            "prof": "Professor", "prof.": "Professor", "professor": "Professor",
            "sir": "Sir", "madam": "Madam", "madame": "Madame",
            "rev": "Reverend", "rev.": "Reverend", "reverend": "Reverend",
            "capt": "Captain", "capt.": "Captain", "captain": "Captain"
        }

        intro_patterns = [
            "my name is ", "i am ", "i'm ", "this is ", "call me ",
            "hi i am ", "hi i'm ", "hello i am ", "hello i'm ",
            "hey i am ", "hey i'm ", "hi my name is ", "hello my name is ",
            "hey my name is ", "myself ", "hi myself", "hey myself", "hello myself", "this is", "here is", "meet", "introducing",
            "let me introduce", "allow me to introduce", "here we have with us", "let me introduce you to", 
            "please welcome today's personality", "say hello to", "allow me to introduce you to", "welcome", "we have here with us"
        ]

        # Pre-processing of text 
        text = input_text.lower().strip()

        # First try to extract name after common introduction patterns
        for pattern in intro_patterns:
            if pattern in text:
                
                name_part = text.split(pattern, 1)[1].strip()
               
              
                tokens = word_tokenize(name_part)
                tagged = pos_tag(tokens)
               
                # Check if first word is a title
                if tokens and tokens[0].lower() in title_expansions:
                    title = title_expansions[tokens[0].lower()]
                   
                    # Get the next word(s) that are proper nouns or just nouns if proper nouns not found
                    proper_nouns = []
                    for token, tag in tagged[1:]:
                        if tag.startswith('NNP') or (not proper_nouns and tag.startswith('NN')):
                            proper_nouns.append(token)
                        elif proper_nouns:
                            
                            break
                   
                    if proper_nouns:
                        
                        name = f"{title} {' '.join(word.capitalize() for word in proper_nouns)}"
                        return name
                    else:
                        # If no name found after title, look for next tokens
                        if len(tokens) > 1:
                            return f"{title} {tokens[1].capitalize()}"
                        return title
               
                # If no title, try to find proper nouns
                chunks = ne_chunk(tagged)
                name_entities = []
               
                # Extract name
                for chunk in chunks:
                    if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                        name_entities.append(' '.join(c[0] for c in chunk))
               
                if name_entities:
                    
                    return name_entities[0].title()
               
                # If NLTK didn't find a person entity, take the first word or words
                if tokens:
                    
                    proper_nouns = []
                    for token, tag in tagged:
                        if tag.startswith('NNP'):
                            proper_nouns.append(token)
                        else:
                            
                            if proper_nouns:
                                break
                   
                    if proper_nouns:
                        return ' '.join(word.capitalize() for word in proper_nouns)
                    else:
                        
                        return tokens[0].capitalize()
       
        # If no intro pattern matched, try direct extraction
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
       
        # Check for standalone titles in the text (e.g., "Dr Smith here to see you")
        for i, (token, tag) in enumerate(tagged):
            if token.lower() in title_expansions and i < len(tagged) - 1:
                title = title_expansions[token.lower()]
                next_token, next_tag = tagged[i+1]
               
                
                if next_tag.startswith('NNP') or next_tag.startswith('NN'):
                    return f"{title} {next_token.capitalize()}"
                else:
                    return title
       
        # If still no match, try NLTK directly on the full text
        chunks = ne_chunk(tagged)
       
        # Extract name
        name_entities = []
        for chunk in chunks:
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                name_entities.append(' '.join(c[0] for c in chunk))
       
        if name_entities:
            return name_entities[0].title()
       
        # If still no name found, look for proper nouns
        proper_nouns = []
        for token, tag in tagged:
            if tag.startswith('NNP'):
                proper_nouns.append(token)
            else:
                
                if proper_nouns:
                    break
       
        if proper_nouns:
            return ' '.join(word.capitalize() for word in proper_nouns)
       
        # Default return if no name found
        return "Friend"

    def generate_hobby_discovery_question(self, name):
        
        return random.choice(self.hobby_questions).format(name=name)

    def generate_initial_greeting(self):
        
        greeting = random.choice(self.greetings)
        name_q = random.choice(self.name_questions)
        return f"{greeting} {name_q}"

    def generate_personalized_questions(self, name):
        
        display_name = name
       
        
        prof_q = random.choice(self.profession_questions).format(name=display_name)
        exp_q = random.choice(self.experience_questions).format(name=display_name)
        hobby_q = self.generate_hobby_discovery_question(display_name)
       
        return prof_q, exp_q, hobby_q
