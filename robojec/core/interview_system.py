import threading
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import random
from anthropic import Anthropic

from robojec.core.followup_generator import FollowUpGenerator
from robojec.core.question_generator import PersonalityQuestionsGenerator
from robojec.core.samvad import SamvadGenerator
from robojec.core.willingness_analyzer import WillingnessAnalyzer, WillingnessLevel


class PersonalityInterviewSystem:
    """
    Central coordinator for the interview session.
    """

    def __init__(
        self,
        client: Anthropic,
        questions_dir: Path,
        main_category: str,
        subcategory: str,
        hobby: str = "",
        user_context: str = "professional",
        user_field: str = "",
    ) -> None:
        self.question_generator   = PersonalityQuestionsGenerator(client, questions_dir)
        self.followup_generator   = FollowUpGenerator(client)
        self.willingness_analyzer = WillingnessAnalyzer()
        self.conversation_starter = SamvadGenerator()

        self.categories: Dict[str, str] = {
            "main":        main_category,
            "subcategory": subcategory,
            "hobby":       hobby if hobby is not None else "",
        }

        # user context for question generation
        self.user_context = user_context
        self.user_field   = user_field

        self.asked_questions: set = set()
        self.lock                 = threading.Lock()
        self.current_willingness  = WillingnessLevel.MEDIUM

        self.available_datasets   = self._check_datasets()
        self.all_questions        = self._preload_questions()

        self.category_question_counts: Dict[str, int] = {
            "main": 0, "subcategory": 0, "hobby": 0
        }

    # ── dataset tracking ───────────────────────────────────────────────────────

    def _check_datasets(self) -> Dict[str, bool]:
        available = {}
        for cat_type, cat_name in self.categories.items():
            if cat_type == "hobby" and not cat_name:
                available[cat_type] = False
                continue
            fp = self.question_generator._file_path(cat_type, cat_name)
            available[cat_type] = fp.exists()
        return available

    def _preload_questions(self) -> set:
        all_q: set = set()
        for cat_type, cat_name in self.categories.items():
            if not cat_name:
                continue
            if not self.available_datasets.get(cat_type, False):
                defaults = self.question_generator._get_defaults(cat_type, cat_name, 10)
                all_q.update(q["question"] for q in defaults if "question" in q)
                continue
            for level in WillingnessLevel:
                questions = self.question_generator.get_questions(
                    cat_type, cat_name, num_questions=25, willingness_level=level
                )
                all_q.update(q["question"] for q in questions if "question" in q)
        return all_q

    def update_available_datasets(self, category_type: str) -> None:
        with self.lock:
            self.available_datasets[category_type] = True
            print(f"  [System] Dataset now available: {category_type}")

    def refresh_available_datasets(self) -> None:
        with self.lock:
            self.available_datasets = self._check_datasets()

    # ── question retrieval ─────────────────────────────────────────────────────

    def get_question_by_category(
        self,
        category_type: str,
        willingness_level: Optional[WillingnessLevel] = None,
    ) -> Dict:
        if willingness_level is None:
            willingness_level = self.current_willingness

        cat_name = self.categories[category_type]
        if not cat_name:
            return self._default_question(willingness_level, category_type)

        if not self.available_datasets.get(category_type, False):
            fp = self.question_generator._file_path(category_type, cat_name)
            if fp.exists():
                self.update_available_datasets(category_type)

        if not self.available_datasets.get(category_type, False):
            return self._default_question(willingness_level, category_type)

        all_levels  = list(WillingnessLevel)
        start_index = all_levels.index(willingness_level)

        for i in range(len(all_levels)):
            level     = all_levels[(start_index + i) % len(all_levels)]
            questions = self.question_generator.get_questions(
                category_type, cat_name,
                num_questions=25,
                willingness_level=level,
                context=self.user_context,
                field=self.user_field,
            )
            new_qs = [q for q in questions if q["question"] not in self.asked_questions]
            if new_qs:
                selected = random.choice(new_qs)
                self.asked_questions.add(selected["question"])
                self.category_question_counts[category_type] += 1
                return {
                    "question_text":    selected["question"],
                    "category_type":    category_type,
                    "category_name":    cat_name,
                    "willingness_level": selected["willingness_level"],
                }

        return self._default_question(willingness_level, category_type)

    def get_hobby_question(
        self, willingness_level: Optional[WillingnessLevel] = None
    ) -> Optional[Dict]:
        if willingness_level is None:
            willingness_level = self.current_willingness

        hobby = self.categories["hobby"]
        if not hobby:
            return None

        if not self.available_datasets.get("hobby", False):
            fp = self.question_generator._file_path("hobby", hobby)
            if fp.exists():
                self.update_available_datasets("hobby")

        if not self.available_datasets.get("hobby", False):
            return None

        questions = self.question_generator.get_questions(
            "hobby", hobby, num_questions=25, willingness_level=willingness_level
        )
        new_qs = [q for q in questions if q["question"] not in self.asked_questions]
        if new_qs:
            selected = random.choice(new_qs)
            self.asked_questions.add(selected["question"])
            self.category_question_counts["hobby"] += 1
            return {
                "question_text":    selected["question"],
                "category_type":    "hobby",
                "category_name":    hobby,
                "willingness_level": selected["willingness_level"],
            }
        return None

    def create_question_from_template(self, template: Dict) -> Dict:
        cat_type = template.get("category_type", "hobby")
        return {
            "question_text":    template["question"],
            "category_type":    cat_type,
            "category_name":    self.categories.get(cat_type, ""),
            "willingness_level": template.get("willingness_level", "medium_willingness"),
        }

    # ── default question fallback ──────────────────────────────────────────────

    def _default_question(
        self,
        willingness_level: WillingnessLevel,
        category_type: Optional[str] = None,
    ) -> Dict:
        if category_type is None:
            non_empty = [ct for ct, cn in self.categories.items() if cn]
            category_type = random.choice(non_empty) if non_empty else "main"

        cat_name = self.categories.get(category_type, "")
        if not cat_name:
            return {
                "question_text":    "Could you tell me more about your interests and experiences?",
                "category_type":    category_type,
                "category_name":    "general",
                "willingness_level": willingness_level.value,
            }

        defaults = self.question_generator._get_defaults(category_type, cat_name, 25)
        if not defaults:
            return {
                "question_text":    f"Could you tell me more about your experience with {cat_name}?",
                "category_type":    category_type,
                "category_name":    cat_name,
                "willingness_level": willingness_level.value,
            }

        matching = [
            q for q in defaults
            if q.get("willingness_level") == willingness_level.value
        ] or defaults

        new_qs   = [q for q in matching if q["question"] not in self.asked_questions]
        selected = random.choice(new_qs) if new_qs else random.choice(matching)
        self.asked_questions.add(selected["question"])
        self.category_question_counts[category_type] += 1

        return {
            "question_text":    selected["question"],
            "category_type":    category_type,
            "category_name":    cat_name,
            "willingness_level": selected["willingness_level"],
        }

    # ── willingness update ─────────────────────────────────────────────────────

    def update_willingness_level(
        self, audio_data: np.ndarray
    ) -> tuple[WillingnessLevel, float]:
        if audio_data is None or len(audio_data) == 0:
            return WillingnessLevel.LOW, 0.0
        level, score, _ = self.willingness_analyzer.analyze_audio_data(audio_data)
        self.current_willingness = level
        return level, score