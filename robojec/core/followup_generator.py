import re
from collections import deque
from typing import List, Optional

from anthropic import Anthropic

from config import CLAUDE_MODEL


class FollowUpGenerator:
    """
    Generates a single contextual follow-up question after a candidate's
    response, using a short rolling context window.
    """

    _BLACKLIST_STARTS = (
        "can you", "could you", "would you",
        "do you", "did you", "have you",
        "is there", "are there", "will you",
    )

    _PROMPT_TEMPLATE = (
        "Generate one 8-12 word follow-up question based on the conversation context below.\n"
        "The question should be natural, relevant, and help explore the candidate's personality "
        "and experiences more deeply.\n\n"
        "Recent Conversation Context:\n{context}\n\n"
        "Guidelines:\n"
        "1. Focus on the most recent response but consider the full context\n"
        "2. Ask about specific details mentioned (people, projects, challenges)\n"
        "3. Explore motivations, feelings, or lessons learned\n"
        "4. Keep it conversational and natural\n"
        "5. Avoid yes/no questions\n"
        "6. Make it personal but professional\n\n"
        "Generate just one follow-up question (8-12 words):"
    )

    def __init__(self, client: Anthropic) -> None:
        self.client          = client
        self.context_history = deque(maxlen=3)
        self.last_followup: Optional[str] = None

    def generate_follow_up(
        self,
        candidate_response: str,
        keywords: List[str],
        word_count: int,
        is_short: bool,
        themes: List[str],
        missing_star: List[str],
    ) -> Optional[str]:
        """
        Return a follow-up question string, or None if conditions aren't met.

        Conditions for generation:
        - Response must be at least 15 words
        - Must not repeat the previous follow-up
        """
        if not candidate_response or len(candidate_response.split()) < 15:
            return None

        self.context_history.append(candidate_response)
        context = "\n".join(f"- {r}" for r in self.context_history)

        try:
            response = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": self._PROMPT_TEMPLATE.format(context=context),
                }],
            )
            question = response.content[0].text.strip()
            # strip leading numbering / quotes
            question = re.sub(r'^[\d."\']+\s*', "", question).strip()

            if not self._is_valid(question):
                return None
            if question == self.last_followup:
                return None

            self.last_followup = question
            return question

        except Exception as exc:
            print(f"  [FollowUp] Generation error: {exc}")
            return None

    # ── validation ─────────────────────────────────────────────────────────────

    def _is_valid(self, question: str) -> bool:
        if not question or not question.endswith("?"):
            return False

        words = question.split()
        if not (8 <= len(words) <= 12):
            return False

        q_lower = question.lower()
        if any(q_lower.startswith(start) for start in self._BLACKLIST_STARTS):
            return False

        return True