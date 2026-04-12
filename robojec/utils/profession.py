"""
Profession recognition — Claude-powered with NLTK fallback.

Claude decides:
- Whether a profession needs a specialisation or IS the specialisation
- What clarification question to ask if needed
- Generates field-specific examples if person is stuck
"""

import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = None

for _res, _path in [
    ("punkt",                      "tokenizers/punkt"),
    ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
    ("stopwords",                  "corpora/stopwords"),
    ("wordnet",                    "corpora/wordnet"),
]:
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_res, quiet=True)


# ── professions that ARE their own specialisation ─────────────────────────────
# No clarification needed for these — the role is specific enough on its own.
_SELF_COMPLETE = {
    # creative / performance
    "actor", "actress", "musician", "singer", "dancer", "writer", "author",
    "poet", "photographer", "filmmaker", "director", "producer",
    # food & hospitality
    "chef", "baker", "barista", "bartender",
    # government & public service
    "politician", "minister", "senator", "governor", "mayor", "diplomat",
    "ambassador", "councillor", "official",
    # skilled trades
    "carpenter", "electrician", "plumber", "mechanic", "tailor",
    "farmer", "fisherman", "gardener",
    # professional services (no sub-specialisation needed)
    "architect", "accountant", "auditor", "economist", "librarian",
    "social worker",
    # health (roles complete without a specialisation)
    "nurse", "paramedic", "pharmacist", "dentist", "veterinarian",
    "psychologist", "therapist",
    # public safety
    "firefighter", "police officer", "soldier", "pilot", "athlete",
    # education (teacher is self-complete — subject is optional)
    "teacher",
    # journalism
    "journalist",
}

# ── umbrella roles that always need a specialisation ─────────────────────────
_NEEDS_SPECIALISATION = {
    "engineer", "doctor", "physician", "surgeon", "scientist", "researcher",
    "developer", "programmer", "designer", "manager", "consultant", "analyst",
    "specialist", "student", "professor", "lecturer",
}


# ── Claude prompt ──────────────────────────────────────────────────────────────

_PROFESSION_PROMPT = """You are helping a conversational AI understand who it is speaking with.

The person said: "{text}"

Respond with ONLY valid JSON — no explanation, no markdown:

{{
  "context": "student | professional | researcher | other",
  "field": "specific field/specialisation — e.g. Machine Learning, Civil Engineering, Cardiology, Criminal Law. Empty string if not mentioned or if profession needs no specialisation.",
  "role": "the role — e.g. ML Student, Cardiologist, Civil Engineer, Politician",
  "main_category": "one of: Technology | Education | Healthcare | Business | Legal | Creative | Science | Manufacturing | Finance | Politics | Other",
  "subcategory": "clean readable title — e.g. Machine Learning Student, Senior Civil Engineer. NO seniority prefixes.",
  "years_context": "studying | working | practising | researching | other",
  "needs_specialisation": true or false,
  "is_complete": true or false,
  "needs_clarification": null,
  "title_prefix": "Dr. | Prof. | null"
}}

RULES:
needs_specialisation = true if the role is a broad umbrella that has meaningful sub-fields (engineer, doctor, scientist, developer, designer, student, manager, researcher, consultant, analyst, professor, surgeon, programmer).
needs_specialisation = false if the role IS its own specialisation (politician, actor, chef, journalist, pilot, nurse, dentist, accountant, teacher, architect, farmer, etc.).

is_complete = true ONLY if:
  - needs_specialisation is false (profession is self-complete), OR
  - needs_specialisation is true AND a specific field/specialisation is clearly mentioned

is_complete = false if needs_specialisation is true but no specific field was mentioned.

When is_complete = false, needs_clarification must be a natural question asking for their specialisation.
When is_complete = true, needs_clarification must be null.

title_prefix: set "Dr." if they are a medical doctor, PhD holder, or mentioned dr./doctor/phd. Set "Prof." if professor. Otherwise null.

field: if needs_specialisation is false, field can be the profession itself or empty.
subcategory must NOT start with Junior/Senior/Experienced/Veteran/Aged.

Return ONLY the JSON object.
"""

_EXAMPLES_PROMPT = """The person said they are a {profession} but couldn't specify their area.
Generate 4-5 common specialisations or sub-fields for {profession} as a short helpful list.
Format: a natural sentence like "For example, fields like X, Y, Z, or W."
Keep it brief, friendly, and genuinely helpful. Return only the sentence."""


# ── Claude helpers ─────────────────────────────────────────────────────────────

def _claude_call(client: Any, prompt: str, max_tokens: int = 400) -> Optional[str]:
    try:
        resp = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as exc:
        print(f"  [Claude] Call failed: {exc}")
        return None


def generate_specialisation_examples(client: Any, profession: str) -> str:
    """Ask Claude to generate helpful examples for a vague profession."""
    if client is None:
        return ""
    result = _claude_call(
        client,
        _EXAMPLES_PROMPT.format(profession=profession),
        max_tokens=80,
    )
    return result or ""


# ── main recognition ───────────────────────────────────────────────────────────

# Adverbs / time phrases that confuse context detection — strip before sending to Claude
_CONTEXT_NOISE = [
    "currently ", "currently, ", "at the moment ", "right now ",
    "presently ", "these days ", "at present ", "today ",
    "for now ", "as of now ", "i am currently ", "i'm currently ",
]


def _clean_profession_text(text: str) -> str:
    """Remove time/context adverbs that confuse student vs professional detection."""
    cleaned = text.lower().strip()
    for noise in _CONTEXT_NOISE:
        cleaned = cleaned.replace(noise, "")
    return cleaned.strip()


# ── years_context correction ──────────────────────────────────────────────────
# Maps professions/categories to the correct verb for "how long have you been X"


_PRACTISING_CATEGORIES = {"Healthcare", "Legal"}
_PRACTISING_ROLES      = {
    "doctor", "physician", "surgeon", "dentist", "pharmacist",
    "therapist", "clinician", "nurse", "lawyer", "attorney",
    "barrister", "solicitor", "advocate", "counsel",
}
_RESEARCHING_ROLES = {
    "researcher", "scientist", "academic", "fellow",
}
_WORKING_CATEGORIES = {
    "Politics", "Business", "Technology", "Creative",
    "Manufacturing", "Finance", "Other",
}


def _correct_years_context(
    years_context: str,
    context: str,
    main_category: str,
    role: str,
) -> str:
    """
    Correct Claude's years_context value.
    Rules:
    - student → always "studying"
    - medical/legal roles → "practising"
    - researchers/scientists → "researching"
    - everything else → "working"
    """
    if context == "student":
        return "studying"

    role_lower = role.lower()

    if any(r in role_lower for r in _PRACTISING_ROLES):
        return "practising"

    if any(r in role_lower for r in _RESEARCHING_ROLES):
        return "researching"

    if main_category in _PRACTISING_CATEGORIES:
        return "practising"

    # government, politics, business, tech, creative — all "working"
    return "working"


def recognize_profession_claude(
    client: Any,
    profession_text: str,
    years_experience: Optional[float] = None,
) -> Dict[str, Any]:
    cleaned = _clean_profession_text(profession_text)
    raw = _claude_call(client, _PROFESSION_PROMPT.format(text=cleaned))
    if raw is None:
        return recognize_profession_fallback(profession_text, years_experience)

    try:
        raw   = re.sub(r"```json|```", "", raw).strip()
        data  = json.loads(raw)

        # enforce our own rules on top of Claude's response
        needs_spec = data.get("needs_specialisation", True)
        field      = data.get("field", "").strip()

        # Generic/umbrella fields — NOT specific enough on their own
        _GENERIC_FIELDS = {
            "education", "medicine", "science", "engineering", "business",
            "technology", "law", "arts", "healthcare", "politics",
            "research", "general", "various", "other", "",
        }
        field_is_specific = field and field.lower() not in _GENERIC_FIELDS

        # ── Deterministic is_complete — ignore Claude's needs_specialisation ──
        # Claude is unreliable about needs_specialisation (calls student self-complete).
        # We decide entirely from our own lists + field check.
        text_lower = cleaned.lower()

        # Step 1: does the text contain a self-complete profession?
        text_is_self_complete = any(p in text_lower for p in _SELF_COMPLETE)

        # Step 2: does the text contain an umbrella role that needs a field?
        text_needs_spec = any(p in text_lower for p in _NEEDS_SPECIALISATION)

        if text_is_self_complete and not text_needs_spec:
            # unambiguously self-complete (actor, chef, pilot...)
            data["is_complete"]         = True
            data["needs_clarification"] = None
            if not field_is_specific:
                data["field"] = data.get("role", "").strip()

        elif text_needs_spec and not field_is_specific:
            # umbrella role (student, engineer, doctor...) with no specific field
            data["is_complete"] = False
            data["field"]       = ""
            data["needs_clarification"] = _fallback_clarification(profession_text)

        elif field_is_specific:
            # specific field present → complete regardless of role type
            data["is_complete"]         = True
            data["needs_clarification"] = None

        elif text_is_self_complete:
            # self-complete even if umbrella also matched (e.g. "teacher and researcher")
            data["is_complete"]         = True
            data["needs_clarification"] = None

        else:
            # unknown profession — ask for clarification to be safe
            data["is_complete"] = False
            data["needs_clarification"] = (
                data.get("needs_clarification")
                or _fallback_clarification(profession_text)
            )

        data["years_experience"] = years_experience
        data["seniority"]        = _seniority_label(
            years_experience, data.get("context", "professional")
        )
        data["interdisciplinary"] = False
        data["_source"]           = "claude"
        data["title_prefix"]      = data.get("title_prefix") or None

        # correct years_context — Claude sometimes returns wrong value
        data["years_context"] = _correct_years_context(
            data.get("years_context", "working"),
            data.get("context", "professional"),
            data.get("main_category", ""),
            data.get("role", ""),
        )

        return data

    except Exception as exc:
        print(f"  [Profession] JSON parse failed: {exc} — using fallback.")
        return recognize_profession_fallback(profession_text, years_experience)


def recognize_profession(
    profession_text: str,
    years_experience: Optional[float] = None,
    client: Any = None,
) -> Dict[str, Any]:
    if client is not None:
        return recognize_profession_claude(client, profession_text, years_experience)
    return recognize_profession_fallback(profession_text, years_experience)


# ── display name helper ────────────────────────────────────────────────────────

def build_display_name(name: str, profession_categories: Dict) -> str:
    """
    Returns name with title prefix if applicable.
    e.g. "Devarshi" → "Dr. Devarshi" if they are a doctor/PhD
    Only adds prefix if person hasn't already introduced themselves with one.
    """
    prefix = profession_categories.get("title_prefix")
    if not prefix:
        return name
    # don't double-prefix
    for p in ["Dr.", "Prof.", "Dr", "Prof"]:
        if name.startswith(p):
            return name
    return f"{prefix} {name}"


# ── seniority ──────────────────────────────────────────────────────────────────

def _seniority_label(years: Optional[float], context: str) -> str:
    if years is None:
        return "mid"
    if context == "student":
        if years <= 1:  return "first year"
        if years <= 2:  return "second year"
        if years <= 3:  return "third year"
        return "final year"
    else:
        if years < 2:   return "junior"
        if years < 5:   return "mid-level"
        if years < 10:  return "senior"
        return "veteran"


def _fallback_clarification(text: str) -> str:
    t = text.lower()
    if "engineer" in t:    return "What type of engineering do you work in?"
    if "doctor"   in t:    return "What is your medical specialisation?"
    if "surgeon"  in t:    return "What type of surgery do you specialise in?"
    if "student"  in t:    return "What are you studying?"
    if "teacher"  in t or "professor" in t: return "What subject do you teach?"
    if "developer" in t or "programmer" in t: return "What kind of development do you focus on?"
    if "designer"  in t:   return "What kind of design do you work on?"
    if "scientist" in t or "researcher" in t: return "What field are you in?"
    if "manager"   in t:   return "What area do you manage?"
    if "consultant" in t:  return "What field do you consult in?"
    if "analyst"   in t:   return "What kind of analysis do you do?"
    return "Could you tell me a bit more about your specific area?"


# ── rule-based fallback ────────────────────────────────────────────────────────

class _FallbackRecognizer:

    _INDUSTRIES = {
        "Technology": {
            "roles":  ["developer", "engineer", "programmer", "architect", "scientist",
                       "analyst", "specialist", "consultant", "technician", "expert",
                       "researcher", "designer"],
            "fields": ["software", "hardware", "data", "network", "cloud", "web", "mobile",
                       "security", "ai", "ml", "machine learning", "artificial intelligence",
                       "deep learning", "computer vision", "nlp", "data science",
                       "mechanical engineering", "civil engineering", "computer science"],
            "terms":  ["code", "programming", "algorithm", "python", "java", "javascript",
                       "tensorflow", "pytorch", "devops", "api", "database"],
        },
        "Education": {
            "roles":  ["professor", "teacher", "lecturer", "instructor", "educator",
                       "tutor", "student", "researcher", "scholar", "faculty", "dean"],
            "fields": ["education", "teaching", "academic", "learning", "curriculum",
                       "school", "university", "college"],
            "terms":  ["study", "studying", "degree", "thesis", "course", "semester",
                       "graduate", "undergraduate", "phd", "masters", "bachelor"],
        },
        "Healthcare": {
            "roles":  ["doctor", "physician", "surgeon", "nurse", "therapist",
                       "specialist", "clinician", "practitioner", "dentist", "pharmacist"],
            "fields": ["medicine", "medical", "clinical", "health", "therapy",
                       "nursing", "surgery", "diagnosis", "cardiology", "neurology"],
            "terms":  ["patient", "hospital", "clinic", "treatment", "prescription"],
        },
        "Business": {
            "roles":  ["manager", "executive", "director", "analyst", "consultant",
                       "coordinator", "advisor", "officer", "accountant"],
            "fields": ["business", "management", "marketing", "sales", "finance",
                       "operations", "strategy", "consulting", "accounting"],
            "terms":  ["corporate", "revenue", "budget", "client", "market"],
        },
        "Legal": {
            "roles":  ["lawyer", "attorney", "counsel", "judge", "paralegal",
                       "advocate", "solicitor", "barrister"],
            "fields": ["law", "legal", "justice", "litigation", "compliance"],
            "terms":  ["case", "court", "contract", "statute", "regulation"],
        },
        "Creative": {
            "roles":  ["designer", "artist", "writer", "director", "producer",
                       "editor", "animator", "actor", "photographer", "musician",
                       "chef", "journalist"],
            "fields": ["design", "art", "media", "content", "creative",
                       "entertainment", "film", "music", "culinary", "journalism"],
            "terms":  ["graphic", "visual", "illustration", "animation", "brand"],
        },
        "Science": {
            "roles":  ["scientist", "researcher", "analyst", "technician", "fellow"],
            "fields": ["research", "science", "laboratory", "biology", "chemistry",
                       "physics", "environmental"],
            "terms":  ["experiment", "hypothesis", "publication", "lab", "grant"],
        },
        "Finance": {
            "roles":  ["analyst", "banker", "advisor", "accountant", "auditor",
                       "trader", "broker", "planner"],
            "fields": ["finance", "investment", "banking", "accounting", "tax",
                       "wealth", "insurance"],
            "terms":  ["portfolio", "stock", "fund", "capital", "revenue"],
        },
        "Politics": {
            "roles":  ["politician", "minister", "senator", "representative",
                       "councillor", "official", "diplomat", "ambassador"],
            "fields": ["politics", "government", "policy", "diplomacy", "public service"],
            "terms":  ["election", "parliament", "congress", "legislation", "governance"],
        },
    }

    _STUDENT_SIGNALS = [
        "student", "studying", "study", "pursuing", "enrolled",
        "undergraduate", "graduate", "postgraduate", "bachelor",
        "master", "phd", "degree", "semester", "college", "university",
    ]

    def __init__(self) -> None:
        self.stop_words = set(stopwords.words("english"))

    def recognize(self, text: str, years: Optional[float] = None) -> Dict[str, Any]:
        processed  = text.lower().strip()
        is_student = any(s in processed for s in self._STUDENT_SIGNALS)
        context    = "student" if is_student else "professional"

        scores: Dict[str, int] = defaultdict(int)
        for industry, defn in self._INDUSTRIES.items():
            for r in defn["roles"]:
                if r in processed: scores[industry] += 3
            for f in defn["fields"]:
                if f in processed: scores[industry] += 2
            for t in defn["terms"]:
                if t in processed: scores[industry] += 1
        if is_student:
            scores["Education"] += 2

        main_category = max(scores, key=scores.get) if scores else "Other"
        field         = self._extract_field(processed)
        role          = "Student" if is_student else self._extract_role(processed, main_category)

        # check self-complete
        is_self_complete = any(p in processed for p in _SELF_COMPLETE)
        needs_spec       = not is_self_complete and any(
            p in processed for p in _NEEDS_SPECIALISATION
        )

        is_complete     = is_self_complete or (needs_spec and bool(field)) or (not needs_spec)
        clarification_q = None
        if not is_complete:
            clarification_q = _fallback_clarification(processed)

        subcategory = f"{field} {role}".strip() if field else role

        # title prefix
        title_prefix = None
        if any(t in processed for t in ["dr.", "dr ", "doctor", "physician"]):
            title_prefix = "Dr."
        elif any(t in processed for t in ["prof.", "prof ", "professor"]):
            title_prefix = "Prof."
        elif "phd" in processed or "ph.d" in processed:
            title_prefix = "Dr."

        years_ctx = _correct_years_context(
            "studying" if is_student else "working",
            context, main_category, role
        )

        return {
            "context":              context,
            "field":                field or "",
            "role":                 role,
            "main_category":        main_category,
            "subcategory":          subcategory,
            "years_context":        years_ctx,
            "needs_specialisation": needs_spec,
            "is_complete":          is_complete,
            "needs_clarification":  clarification_q,
            "years_experience":     years,
            "seniority":            _seniority_label(years, context),
            "interdisciplinary":    False,
            "title_prefix":         title_prefix,
            "_source":              "fallback",
        }

    def _extract_field(self, text: str) -> str:
        multi = [
            "machine learning", "artificial intelligence", "deep learning",
            "computer vision", "natural language processing", "data science",
            "computer science", "civil engineering", "mechanical engineering",
            "software engineering", "electrical engineering", "software development",
            "information technology", "cyber security", "web development",
            "criminal law", "corporate law", "family law", "international law",
            "cardiology", "neurology", "orthopaedics", "paediatrics", "oncology",
            "environmental science", "political science", "social science",
        ]
        for f in multi:
            if f in text:
                return f.title()
        for defn in self._INDUSTRIES.values():
            for f in defn["fields"]:
                if f in text and len(f) > 5:
                    return f.title()
        return ""

    def _extract_role(self, text: str, category: str) -> str:
        roles = self._INDUSTRIES.get(category, {}).get("roles", [])
        for role in roles:
            if role in text:
                return role.title()
        for word, tag in nltk.pos_tag(word_tokenize(text)):
            if tag.startswith("NN") and len(word) > 3 and word not in self.stop_words:
                return word.title()
        return "Professional"


def recognize_profession_fallback(
    profession_text: str,
    years_experience: Optional[float] = None,
) -> Dict[str, Any]:
    cleaned = _clean_profession_text(profession_text)
    return _FallbackRecognizer().recognize(cleaned, years_experience)