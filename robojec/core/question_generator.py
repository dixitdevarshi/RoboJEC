import csv
import random
from pathlib import Path
from typing import Dict, List

from anthropic import Anthropic

from config import CLAUDE_MODEL, QUESTIONS_DIR, QUESTIONS_PER_CATEGORY
from robojec.core.willingness_analyzer import WillingnessLevel


class PersonalityQuestionsGenerator:
    """
    Generates and caches personality interview questions per category.

    Questions are stored as CSV files under `questions_dir` so they
    survive across sessions and avoid redundant API calls.
    """

    _DEFAULT_TEMPLATES: Dict[str, List[str]] = {
        "main": [
            "How has {category} changed your daily routine?",
            "What excites you most about {category} today?",
            "Do you see {category} differently than five years ago?",
            "What's the biggest challenge in {category} now?",
            "Has {category} met your expectations so far?",
            "Where do you see {category} heading next?",
            "Is there something about {category} people misunderstand?",
            "How has {category} impacted your career path?",
            "What {category} trend seems overrated to you?",
            "Do you think {category} is accessible to everyone?",
            "What {category} skill seems most valuable today?",
            "Has {category} changed how you solve problems?",
            "What would improve {category} for beginners?",
            "Do ethics in {category} get enough attention?",
            "What's your favourite aspect of {category}?",
            "Has {category} connected you with interesting people?",
            "What's one {category} myth you'd like to debunk?",
            "How do you stay current with {category}?",
            "What {category} resource would you recommend?",
            "Has {category} become more complex over time?",
            "Do you think {category} is changing society positively?",
            "What {category} development are you watching closely?",
            "Is {category} headed in the right direction?",
            "What drew you to {category} initially?",
            "How might {category} evolve in five years?",
        ],
        "subcategory": [
            "What's the best part of working in {category}?",
            "Has your perspective on {category} changed over time?",
            "What skill in {category} took longest to develop?",
            "Do people misunderstand what {category} professionals actually do?",
            "What {category} challenge do you face regularly?",
            "Has technology changed how you approach {category}?",
            "What attracted you to {category} initially?",
            "Is work-life balance possible in {category}?",
            "What {category} task do you find most rewarding?",
            "Has {category} become more competitive recently?",
            "What's something about {category} that surprised you?",
            "Do you think {category} gets proper recognition?",
            "What tool or method revolutionised your {category} work?",
            "How do you explain {category} to someone unfamiliar?",
            "What's changing fastest in {category} right now?",
            "Do you mentor others in {category}?",
            "What {category} skill is undervalued today?",
            "Has your definition of success in {category} evolved?",
            "What keeps you motivated in {category}?",
            "Do you collaborate with others in {category}?",
            "What would you change about {category} education?",
            "Has specialising in {category} been worth it?",
            "What's one {category} mistake people often make?",
            "How do you handle stress in {category}?",
            "What makes someone truly excel in {category}?",
        ],
        "hobby": [
            "What first caught your interest in {category}?",
            "What still amazes you about {category}?",
            "Has {category} changed how you see things?",
            "What's your personal style with {category}?",
            "What {category} challenge changed you most?",
            "What has {category} revealed about yourself?",
            "What {category} experience would you share?",
            "Has your approach to {category} evolved?",
            "What {category} question still intrigues you?",
            "Who has {category} connected you with?",
            "Any personal rituals around your {category}?",
            "Does your mood affect your {category} practice?",
            "What's something hard about {category} for you?",
            "Has {category} affected your relationships?",
            "What aspect of {category} challenges you most?",
            "How has {category} influenced your space?",
            "What's your guiding principle with {category}?",
            "Has {category} been healing for you?",
            "What {category} myth have you disproven?",
            "Has your background influenced your {category}?",
            "Do you set limits around {category}?",
            "Does perfectionism affect your {category}?",
            "How would you map your {category} journey?",
            "What values has {category} reinforced?",
            "What's your most treasured {category} memory?",
        ],
    }

    _PROMPTS = {
        "main": (
            "Generate {n} unique conversational questions about the personality trait: {category}.\n\n"
            "{context_note}"
            "REQUIREMENTS:\n"
            "- Questions MUST be 8-15 words each\n"
            "- Conversational — sound like two people talking, not an interviewer\n"
            "- Respectful but not overly sophisticated\n"
            "- Each question must explore a COMPLETELY DIFFERENT aspect of {category}\n"
            "- Avoid ANY semantic overlap between questions\n"
            "- Include personal touches like 'How do you...' or 'What's your...'\n"
            "- NO lengthy setups or explanations\n\n"
            "FORMAT: One short question per line, no numbering."
        ),
        "subcategory": (
            "Generate {n} unique conversational questions for someone who is: {category}.\n\n"
            "{context_note}"
            "REQUIREMENTS:\n"
            "- Questions MUST be 8-15 words only\n"
            "- Sound like two people conversing, not an interview\n"
            "- Direct and genuine — as if in a thoughtful dialogue\n"
            "- Adapt tone to context: student questions about learning/studying, "
            "professional questions about work/career\n"
            "- Each explores a DIFFERENT aspect of their experience\n"
            "- No semantic overlap\n"
            "- NO lengthy setups\n\n"
            "BAD (too formal): 'What educational trajectory would you recommend for aspiring professionals?'\n"
            "GOOD (natural): 'What drew you to studying {category}?' or 'What's the hardest part of {category}?'\n\n"
            "FORMAT: One short question per line, no numbering."
        ),
        "hobby": (
            "Generate {n} unique conversational questions about the hobby: {category}.\n\n"
            "{context_note}"
            "REQUIREMENTS:\n"
            "- Questions MUST be 8-15 words only\n"
            "- Sound like two people conversing — direct and genuine\n"
            "- Each explores a DIFFERENT aspect of {category}\n"
            "- No semantic overlap\n"
            "- NO lengthy setups\n\n"
            "BAD: 'What initially catalyzed your interest in pursuing {category} as a hobby?'\n"
            "GOOD: 'What first drew you to {category}?'\n\n"
            "FORMAT: One short question per line, no numbering."
        ),
    }

    def __init__(self, client: Anthropic, questions_dir: Path) -> None:
        self.client        = client
        self.questions_dir = Path(questions_dir)
        self.questions_dir.mkdir(parents=True, exist_ok=True)
        self._used_defaults: set = set()

    # ── public API ─────────────────────────────────────────────────────────────

    def get_questions(
        self,
        category_type: str,
        category: str,
        num_questions: int = QUESTIONS_PER_CATEGORY,
        willingness_level: WillingnessLevel = WillingnessLevel.MEDIUM,
        context: str = "professional",
        field: str = "",
    ) -> List[Dict]:
        file_path = self._file_path(category_type, category)

        if file_path.exists():
            with open(file_path, newline="") as fh:
                rows = list(csv.DictReader(fh))
            print(f"  [QGen] Loaded {len(rows)} questions for {category_type}:{category}")
        else:
            print(f"  [QGen] Generating questions for {category_type}:{category} …")
            rows = self._generate(category_type, category, QUESTIONS_PER_CATEGORY,
                                  context=context, field=field)

        if not rows:
            return self._get_defaults(category_type, category, num_questions)

        filtered = [q for q in rows if q.get("willingness_level") == willingness_level.value]
        pool     = filtered if len(filtered) >= num_questions else rows

        if not pool:
            return self._get_defaults(category_type, category, num_questions)

        return random.sample(pool, min(num_questions, len(pool)))

    def _file_path(self, category_type: str, category: str) -> Path:
        safe = category.lower().replace(" ", "_")
        return self.questions_dir / f"{category_type}_{safe}_questions.csv"

    # ── default questions (no API call) ───────────────────────────────────────

    def _get_defaults(
        self, category_type: str, category: str, num_questions: int = 25
    ) -> List[Dict]:
        templates = self._DEFAULT_TEMPLATES.get(
            category_type, self._DEFAULT_TEMPLATES["main"]
        )
        available = [t for t in templates if t.format(category=category) not in self._used_defaults]
        if not available:
            return []

        selected = available[:num_questions]
        thirds   = max(len(selected) // 3, 1)
        results  = []

        for i, tmpl in enumerate(selected):
            question = tmpl.format(category=category)
            self._used_defaults.add(question)
            willingness = (
                "low_willingness"    if i < thirds
                else "medium_willingness" if i < 2 * thirds
                else "high_willingness"
            )
            results.append({
                "question":         question,
                "category_type":    category_type,
                "category":         category,
                "willingness_level": willingness,
            })

        return results

    # ── LLM generation ─────────────────────────────────────────────────────────

    def _generate(
        self, category_type: str, category: str, num_questions: int,
        context: str = "professional", field: str = ""
    ) -> List[Dict]:
        try:
            questions = self._call_api(category_type, category, num_questions, context, field)
            questions = self._deduplicate(questions, num_questions, category_type, category, context, field)
            questions = [q for q in questions if 8 <= len(q.split()) <= 15]
            questions.sort(key=lambda q: len(q.split()))

            thirds    = max(len(questions) // 3, 1)
            processed = []
            for i, q in enumerate(questions[:num_questions]):
                willingness = (
                    "low_willingness"    if i < thirds
                    else "medium_willingness" if i < 2 * thirds
                    else "high_willingness"
                )
                processed.append({
                    "question":         q,
                    "category_type":    category_type,
                    "category":         category,
                    "willingness_level": willingness,
                })

            if processed:
                self._save(processed, category_type, category)

            return processed

        except Exception as exc:
            print(f"  [QGen] Generation error: {exc}")
            return []

    def _call_api(self, category_type: str, category: str, n: int,
                  context: str = "professional", field: str = "") -> List[str]:
        context_note = ""
        if context == "student":
            context_note = (
                f"IMPORTANT CONTEXT: This person is a STUDENT studying {field or category}. "
                "Questions should be about their learning journey, studies, and academic experience — "
                "NOT about work or career.\n\n"
            )
        elif field:
            context_note = (
                f"IMPORTANT CONTEXT: This person works in {field}. "
                "Questions should reflect their professional experience.\n\n"
            )

        prompt   = self._PROMPTS.get(category_type, self._PROMPTS["main"])
        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt.format(
                n=n, category=category, context_note=context_note
            )}],
        )
        return [
            line.strip()
            for line in response.content[0].text.split("\n")
            if line.strip() and "?" in line
        ]

    def _deduplicate(
        self, questions: List[str], target: int, category_type: str, category: str,
        context: str = "professional", field: str = ""
    ) -> List[str]:
        unique = list(set(questions))
        if len(unique) >= target:
            return unique

        # one extra call to fill the gap
        try:
            context_note = ""
            if context == "student":
                context_note = (
                    f"IMPORTANT CONTEXT: This person is a STUDENT studying {field or category}. "
                    "Questions should be about their learning journey — NOT about work.\n\n"
                )
            elif field:
                context_note = (
                    f"IMPORTANT CONTEXT: This person works in {field}.\n\n"
                )
            extra_prompt = (
                self._PROMPTS.get(category_type, self._PROMPTS["main"]).format(
                    n=target, category=category, context_note=context_note
                )
                + "\n\nIMPORTANT: These must be COMPLETELY DIFFERENT from:\n"
                + "\n".join(unique[:10])
            )
            response = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4000,
                messages=[{"role": "user", "content": extra_prompt}],
            )
            extras = [
                line.strip()
                for line in response.content[0].text.split("\n")
                if line.strip() and "?" in line
            ]
            unique = list(set(unique + self._filter_similar(extras, unique)))
        except Exception as exc:
            print(f"  [QGen] Dedup extra call failed: {exc}")

        return unique

    @staticmethod
    def _filter_similar(candidates: List[str], existing: List[str]) -> List[str]:
        _STOP = {"about", "would", "could", "think", "there", "their", "where",
                 "when", "what", "that", "have", "your", "with", "this"}
        filtered = []
        for candidate in candidates:
            cw = [w for w in candidate.lower().split() if len(w) > 4 and w not in _STOP]
            similar = False
            for exist in existing:
                ew = [w for w in exist.lower().split() if len(w) > 4 and w not in _STOP]
                if sum(1 for w in cw if w in ew) >= 3:
                    similar = True
                    break
            if not similar:
                filtered.append(candidate)
        return filtered

    def _save(self, questions: List[Dict], category_type: str, category: str) -> None:
        path = self._file_path(category_type, category)
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["question", "category_type", "category", "willingness_level"]
            )
            writer.writeheader()
            writer.writerows(questions)
        print(f"  [QGen] Saved {len(questions)} questions → {path}")