import csv
import json
import random
import statistics
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from anthropic import Anthropic

from config import (
    ANTHROPIC_API_KEY,
    FOLLOWUP_MIN_WORDS,
    FOLLOWUP_PROBABILITY,
    HOBBY_QUESTION_COUNT,
    PROFESSIONAL_QUESTION_COUNT,
    QUESTIONS_DIR,
    RESPONSES_DIR,
)
from robojec.core.interview_system import PersonalityInterviewSystem
from robojec.core.question_generator import PersonalityQuestionsGenerator
from robojec.core.willingness_analyzer import WillingnessLevel
from robojec.pipeline.user_info import get_user_info
from robojec.utils.audio import MetaRequest, listen_and_save, listen_and_save_name
from robojec.utils.text_utils import (
    check_star,
    extract_hobbies,
    extract_keywords,
    identify_themes,
)
from robojec.utils.tts import speak, speak_and_record


# ── background dataset generation ─────────────────────────────────────────────

def generate_dataset_background(
    question_generator: PersonalityQuestionsGenerator,
    category_type: str,
    category_name: str,
    system: PersonalityInterviewSystem,
    ready_event: Optional[threading.Event] = None,
) -> None:
    try:
        print(f"  [BG] Generating {category_type}:{category_name} …")
        questions = question_generator._generate(category_type, category_name, 75)
        if questions:
            with threading.Lock():
                for q in questions:
                    system.all_questions.add(q["question"])
            system.update_available_datasets(category_type)
            print(f"  [BG] Done — {category_type}:{category_name}")
        else:
            print(f"  [BG] No questions for {category_type}:{category_name}")
    except Exception as exc:
        print(f"  [BG] Error: {exc}")
    finally:
        if ready_event:
            ready_event.set()


# ── response persistence ───────────────────────────────────────────────────────

def save_response(
    response_dir: Path,
    question: Dict,
    answer: str,
    recording_dir: Path,
    recording_id: str,
    audio_data: Optional[np.ndarray] = None,
    question_audio_file: Optional[Path] = None,
) -> float:
    try:
        word_count     = len(answer.split()) if answer else 0
        estimated_time = word_count / 2.5 if word_count > 0 else 1.0

        precise_time: Optional[float] = None
        if audio_data is not None and hasattr(audio_data, "size") and audio_data.size > 0:
            precise_time = len(audio_data) / 16000

        file_time: Optional[float] = None
        audio_file = recording_dir / f"{recording_id}.wav"
        if audio_file.exists():
            try:
                with wave.open(str(audio_file), "rb") as wf:
                    rate = float(wf.getframerate())
                    if rate > 0:
                        file_time = wf.getnframes() / rate
            except Exception:
                pass

        time_spent = max(0.1, precise_time or file_time or estimated_time)
        timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        record = {
            "timestamp":           timestamp,
            "category_type":       question.get("category_type", "unknown"),
            "category_name":       question.get("category_name", "unknown"),
            "question_text":       question.get("question_text", ""),
            "answer_text":         answer,
            "word_count":          word_count,
            "time_spent":          round(time_spent, 2),
            "willingness_level":   question.get("willingness_level", "medium"),
            "audio_file":          str(audio_file.relative_to(recording_dir.parent)) if audio_file.exists() else None,
            "audio_duration":      round(file_time, 2) if file_time else round(time_spent, 2),
            "question_audio_file": (
                str(question_audio_file.relative_to(recording_dir.parent))
                if question_audio_file and question_audio_file.exists() else None
            ),
        }

        csv_path    = response_dir / "interview_responses.csv"
        file_exists = csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=record.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(record)

        json_path = response_dir / f"{recording_id}_response.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2)

        return time_spent

    except Exception as exc:
        print(f"  [Save] Error: {exc}")
        return 1.0


# ── repeat / rephrase handling ─────────────────────────────────────────────────

def _handle_meta_request(
    meta: MetaRequest,
    question_text: str,
    client: Optional[Anthropic],
) -> None:
    """Handle a repeat request — say the question again."""
    if meta.kind == MetaRequest.REPEAT:
        response = f"Sure. {question_text}"
        print(f"  [Repeat] {response}")
        speak(response)


def _get_rephrased(question_text: str, client: Optional[Anthropic]) -> str:
    """Ask Claude to rephrase a question more simply. Returns original on failure."""
    if client:
        try:
            resp = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=60,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Rephrase this interview question in simpler, clearer language. "
                        f"Plain everyday words. Keep it 8-15 words. "
                        f"Return only the rephrased question:\n{question_text}"
                    )
                }]
            )
            rephrased = resp.content[0].text.strip().strip('"')
            if rephrased and "?" in rephrased:
                return rephrased
        except Exception:
            pass
    return question_text


def _listen_with_meta_handling(
    question_text: str,
    recording_dir: Path,
    question_id: str,
    client: Optional[Anthropic],
    silence_threshold: float = 3.5,
    max_meta_retries: int = 3,
):
    """
    Listen for an answer. If repeat/rephrase requested, handle it and listen again.
    Tracks repeat and rephrase counts separately.
    - Repeat:   say question again, up to 2 times
    - Rephrase: ask Claude to reword, up to 2 times
    - After exhausting both, move on to next question
    Returns (answer_text, audio_data, processing_end_time)
    """
    repeat_count  = 0
    rephrase_count = 0
    # keep track of current question text (may change after rephrase)
    current_question = question_text
    total_meta = 0

    while total_meta <= max_meta_retries:
        result = listen_and_save(
            recording_dir=recording_dir,
            question_id=question_id,
            silence_threshold=silence_threshold,
        )
        answer, audio_data, proc_end = result

        if not isinstance(answer, MetaRequest):
            return answer, audio_data, proc_end

        total_meta += 1

        if answer.kind == MetaRequest.REPEAT and repeat_count < 2:
            _handle_meta_request(answer, current_question, client)
            repeat_count += 1

        elif answer.kind == MetaRequest.REPHRASE and rephrase_count < 2:
            # after rephrase, current_question becomes the rephrased version
            rephrased = _get_rephrased(current_question, client)
            current_question = rephrased
            response = f"Of course, let me put that differently. {rephrased}"
            print(f"  [Rephrase] {response}")
            speak(response)
            rephrase_count += 1

        else:
            # exhausted retries for this type — acknowledge and move on
            msg = "Let's move on to the next question."
            print(f"  [Meta] {msg}")
            speak(msg)
            return "", audio_data, proc_end

    # too many meta requests — move on
    msg = "Let's continue with the interview."
    print(f"  [Meta] {msg}")
    speak(msg)
    return "", np.array([], dtype=np.float32), time.time()


# ── hobby intro ────────────────────────────────────────────────────────────────

def _generate_hobby_intro(
    client: Optional[Anthropic],
    hobby_answer: str,
    display_name: str,
    num_questions: int = 3,
) -> Tuple[str, List[str]]:
    """
    Given the raw hobby answer:
    - Generate a natural transition sentence into Phase 3
    - Extract a clean list of hobbies to cover
    - Distribute questions across hobbies:
        1 hobby  → all questions on that hobby
        2 hobbies → split evenly
        3+ hobbies → pick 2, split evenly

    Returns (transition_sentence, [hobby1, hobby2, ...])
    The returned list has one entry per question slot (may repeat hobbies).
    """
    if client:
        try:
            resp = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=150,
                messages=[{
                    "role": "user",
                    "content": (
                        f"{display_name} said this about their hobbies: \"{hobby_answer}\"\n\n"
                        "Do two things:\n"
                        "1. Write one warm, natural transition sentence moving into hobby questions. "
                        "Sound like a curious person, not a list reader. "
                        "Do NOT list hobbies back verbatim. "
                        "Do NOT mention office, workplace, or work. "
                        "Must work for anyone — student, retiree, professional. "
                        "Examples: \n"
                        "  - 'That sounds like a wonderful mix of interests — I'd love to explore them.' \n"
                        "  - 'How lovely to have such varied passions in life.'\n"
                        "2. List the distinct hobbies as SHORT clean nouns or gerunds (1-3 words each). "
                        "Maximum 3 hobbies. No sentences. No commentary.\n\n"
                        "Respond in EXACTLY this format:\n"
                        "INTRO: <one sentence>\n"
                        "HOBBIES: <hobby1>, <hobby2>"
                    )
                }]
            )
            text     = resp.content[0].text.strip()
            intro    = ""
            hobbies  = []
            for line in text.splitlines():
                if line.startswith("INTRO:"):
                    intro = line.replace("INTRO:", "").strip()
                elif line.startswith("HOBBIES:"):
                    raw = line.replace("HOBBIES:", "").strip()
                    hobbies = [h.strip() for h in raw.split(",") if h.strip()]

            # validate — hobbies must be short (not sentences)
            hobbies = [h for h in hobbies if len(h.split()) <= 4 and "?" not in h]

            if intro and hobbies:
                question_plan = _distribute_hobbies(hobbies, num_questions)
                return intro, question_plan
        except Exception as exc:
            print(f"  [HobbyIntro] Claude failed: {exc}")

    # fallback
    hobbies  = extract_hobbies(hobby_answer)
    hobbies  = [h for h in hobbies if len(h.split()) <= 4][:3]
    if not hobbies:
        hobbies = ["your interests"]
    intro    = f"That's great to hear, {display_name}. Let's explore some of those interests."
    question_plan = _distribute_hobbies(hobbies, num_questions)
    return intro, question_plan


def _distribute_hobbies(hobbies: List[str], num_questions: int) -> List[str]:
    """
    Return a list of length num_questions where each entry is the hobby
    that question slot should focus on.

    1 hobby  → all slots get that hobby
    2 hobbies → alternate: [h1, h2, h1] for 3 questions
    3+ hobbies → pick first 2, alternate
    """
    if len(hobbies) == 1:
        return [hobbies[0]] * num_questions

    # use at most 2 hobbies
    h1, h2 = hobbies[0], hobbies[1]
    plan   = []
    for i in range(num_questions):
        plan.append(h1 if i % 2 == 0 else h2)
    return plan


# ── intelligent question picker ────────────────────────────────────────────────

_QUESTION_PICKER_PROMPT = """You are helping a conversational AI decide what to ask next in a personality interview.

Person info:
- Name: {name}
- Field: {field}
- Role: {role}
- Context: {context} ({seniority})

Conversation so far:
{history}

Candidate question from the pool:
"{candidate}"

Decide: use the candidate as-is, refine it slightly, or generate a better question based on what's been discussed.

Rules:
- Do NOT repeat topics already covered
- Must be about their professional or academic life
- Feel like a natural next step in the conversation
- 8-15 words, warm and curious tone
- Return ONLY the final question, nothing else
"""


def _pick_intelligent_question(
    client: Anthropic,
    candidate: Dict,
    conversation_history: List[Dict],
    user_info: Dict,
    willingness_level: WillingnessLevel,
) -> Dict:
    try:
        prof_cats    = user_info.get("profession_categories", {})
        history_text = "\n".join(
            f"Q: {t['question']}\nA: {t['answer']}"
            for t in conversation_history[-4:]
        )

        resp = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=80,
            messages=[{
                "role": "user",
                "content": _QUESTION_PICKER_PROMPT.format(
                    name=user_info.get("display_name", user_info.get("name", "")),
                    field=prof_cats.get("field", ""),
                    role=prof_cats.get("subcategory", ""),
                    context=prof_cats.get("context", "professional"),
                    seniority=prof_cats.get("seniority", ""),
                    history=history_text,
                    candidate=candidate["question_text"],
                )
            }]
        )

        new_q = resp.content[0].text.strip().strip('"')
        if new_q and "?" in new_q and 5 <= len(new_q.split()) <= 20:
            return {**candidate, "question_text": new_q}

    except Exception as exc:
        print(f"  [QPicker] Failed, using candidate: {exc}")

    return candidate


# ── feedback question ──────────────────────────────────────────────────────────

def ask_feedback_question(
    display_name: str,
    response_dir: Path,
    recording_dir: Path,
    client: Optional[Anthropic] = None,
) -> None:
    feedback_q = _claude_text_local(
        client,
        prompt=(
            f"Generate one closing question to ask {display_name} at the end of a personality interview. "
            "Something reflective — like advice they'd give, or wisdom they'd share. "
            "Warm, professional. 10-18 words. Address them by name. Return only the question."
        ),
        fallback=(
            f"Before we conclude, {display_name}, "
            "what's one piece of advice you'd give to those just starting out in your field?"
        ),
    )

    print(feedback_q)
    feedback_audio_file, _ = speak_and_record(feedback_q, recording_dir, "feedback_question")

    print("\nListening for your answer…")
    feedback_text, feedback_audio = listen_and_save_name(
        recording_dir=recording_dir,
        question_id="feedback",
        silence_threshold=3.0,
        min_speech_duration=1.5,
        max_recording_duration=480,
    )

    if feedback_text and feedback_text.lower() != "quit":
        save_response(
            response_dir,
            {
                "category_type":    "feedback",
                "category_name":    "Interview Experience",
                "question_text":    feedback_q,
                "willingness_level": "medium",
            },
            feedback_text,
            recording_dir,
            "feedback",
            feedback_audio,
            feedback_audio_file,
        )


def _claude_text_local(
    client: Optional[Anthropic], prompt: str,
    max_tokens: int = 80, fallback: str = ""
) -> str:
    if client is None:
        return fallback
    try:
        resp = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip().strip('"')
    except Exception:
        return fallback


# ── timing helpers ─────────────────────────────────────────────────────────────

def _write_timing_csv(
    response_dir: Path,
    timings: List[Tuple],
    all_gaps: List[float],
    prof_gaps: List[float],
    hobby_gaps: List[float],
    interrupted: bool = False,
) -> None:
    suffix = "_interrupted" if interrupted else ""
    path   = response_dir / f"timing_data{suffix}.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Phase", "Event Type", "Event Name", "Timestamp", "Gap"])
        for name, ts in timings:
            phase = (
                "Hobby Discovery" if "discovery" in name
                else "Hobby" if "hobby" in name
                else "Professional"
            )
            etype = (
                "Question Prep"   if "prep" in name
                else "First Byte" if "first_byte" in name
                else "Response End" if "response_end" in name
                else "Gap" if "gap" in name
                else "Other"
            )
            writer.writerow([phase, etype, name, f"{ts:.3f}",
                             f"{ts:.3f}" if "gap" in name else ""])
        writer.writerow([])
        writer.writerow(["Summary"])
        for label, gaps in [("All", all_gaps), ("Professional", prof_gaps), ("Hobby", hobby_gaps)]:
            if not gaps:
                continue
            writer.writerow([label, "avg", f"{statistics.mean(gaps):.3f}"])
            writer.writerow([label, "max", f"{max(gaps):.3f}"])
            writer.writerow([label, "min", f"{min(gaps):.3f}"])


def _print_timing_stats(all_gaps, prof_gaps, hobby_gaps):
    for label, gaps in [("Overall", all_gaps), ("Professional", prof_gaps), ("Hobby", hobby_gaps)]:
        if not gaps:
            continue
        print(f"\n  ⏱ {label} — avg={statistics.mean(gaps):.2f}s  "
              f"max={max(gaps):.2f}s  min={min(gaps):.2f}s")


# ── maybe follow-up ────────────────────────────────────────────────────────────

def _maybe_followup(
    system: PersonalityInterviewSystem,
    answer: str,
    question_count: int,
    recording_dir: Path,
    last_response_end: float,
    prof_gaps: List[float],
    all_gaps: List[float],
    timings: List[Tuple],
    client: Optional[Anthropic],
    question_text: str,
) -> Tuple[str, float, bool]:
    if not (answer and len(answer.split()) > FOLLOWUP_MIN_WORDS
            and random.random() < FOLLOWUP_PROBABILITY):
        return answer, last_response_end, False

    keywords = extract_keywords(answer)
    themes   = identify_themes(answer)
    missing  = check_star(answer)
    substantial = (
        len(keywords) >= 3
        or any(t in themes for t in ["teamwork", "leadership", "challenge", "achievement"])
        or len(missing) >= 2
    )
    if not substantial:
        return answer, last_response_end, False

    follow_up = system.followup_generator.generate_follow_up(
        answer, keywords, len(answer.split()), False, themes, missing
    )
    if not follow_up:
        return answer, last_response_end, False

    print(f"\n  [Follow-up] {follow_up}")
    fu_audio, fu_byte = speak_and_record(
        follow_up, recording_dir, f"q{question_count}_followup_question"
    )

    if last_response_end:
        gap = fu_byte - last_response_end
        prof_gaps.append(gap); all_gaps.append(gap)
        timings.append(("followup_true_gap", gap))

    print("\n[Please respond to follow-up]")
    fu_answer, _, fu_end = _listen_with_meta_handling(
        follow_up, recording_dir, f"q{question_count}_followup", client
    )
    timings.append(("followup_end", fu_end))

    if fu_answer and fu_answer.lower() == "quit":
        return "quit", fu_end, True

    extended = answer + (f"\n\n[Follow-up] {follow_up}\n{fu_answer}" if fu_answer else "")
    return extended, fu_end, True


# ── main interview orchestrator ────────────────────────────────────────────────

def conduct_interview(
    system: PersonalityInterviewSystem,
    user_name: str,
    recording_dir: Path,
    client: Optional[Anthropic] = None,
    user_info: Optional[Dict] = None,
) -> None:
    user_info    = user_info or {}
    display_name = user_info.get("display_name", user_name)

    welcome = _claude_text_local(
        client,
        prompt=(
            f"Generate one warm sentence welcoming {display_name} to the interview "
            "and saying you'll begin now. Professional, not over the top. "
            "Return only the sentence."
        ),
        fallback=f"Wonderful, {display_name}. Let's begin our conversation.",
    )
    print(f"\n{welcome}"); speak(welcome)

    instructions = (
        "I'll ask you some questions to get to know you better. "
        "Feel free to share as much or as little as you'd like."
    )
    print(instructions); speak(instructions)

    response_dir = Path(RESPONSES_DIR) / user_name.lower().replace(" ", "_")
    response_dir.mkdir(parents=True, exist_ok=True)

    start_time        = time.time()
    question_count    = 0
    willingness_level = WillingnessLevel.MEDIUM
    last_was_followup = False
    last_response_end: Optional[float] = None

    prof_gaps:  List[float] = []
    hobby_gaps: List[float] = []
    all_gaps:   List[float] = []
    timings:    List[Tuple] = []

    conversation_history: List[Dict] = []

    try:
        # ══════════════════════════════════════════════════════════════════════
        # PHASE 1 — Professional questions
        # ══════════════════════════════════════════════════════════════════════
        print("\n=== PHASE 1: Professional Experience ===")

        for seq_idx in range(PROFESSIONAL_QUESTION_COUNT):
            question       = system.get_question_by_category("subcategory", willingness_level)
            question_count += 1

            if client and conversation_history and seq_idx > 0:
                question = _pick_intelligent_question(
                    client, question, conversation_history, user_info, willingness_level
                )

            q_prep = time.time()
            timings.append(("question_prep", q_prep))

            print(f"\nQ{question_count}/{PROFESSIONAL_QUESTION_COUNT} [{question['category_name']}]")
            print(f"  {question['question_text']}")

            q_audio, first_byte = speak_and_record(
                question["question_text"], recording_dir,
                f"q{question_count}_prof_question"
            )
            timings.append(("first_byte", first_byte))

            if last_response_end:
                gap = first_byte - last_response_end
                prof_gaps.append(gap); all_gaps.append(gap)
                timings.append(("true_gap", gap))
                print(f"  ⏱ {gap:.2f}s")

            print("\n[Please respond]")
            answer, audio_data, proc_end = _listen_with_meta_handling(
                question["question_text"], recording_dir,
                f"q{question_count}_prof", client
            )
            last_response_end = proc_end
            timings.append(("response_end", proc_end))

            if answer and answer.lower() == "quit":
                _end_interview(display_name); return

            full_answer = answer if answer else "[No response]"

            if not last_was_followup:
                full_answer, last_response_end, last_was_followup = _maybe_followup(
                    system, full_answer, question_count, recording_dir,
                    last_response_end, prof_gaps, all_gaps, timings,
                    client, question["question_text"]
                )
                if full_answer == "quit":
                    _end_interview(display_name); return
            else:
                last_was_followup = False

            save_response(response_dir, question, full_answer,
                          recording_dir, f"q{question_count}_prof",
                          audio_data, q_audio)

            conversation_history.append({
                "question": question["question_text"],
                "answer":   full_answer,
            })

            if audio_data is not None and len(audio_data) > 0:
                willingness_level, _ = system.update_willingness_level(audio_data)

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 2 — Hobby discovery
        # ══════════════════════════════════════════════════════════════════════
        print("\n=== PHASE 2: Personal Interests ===")
        last_response_end = None

        # natural transition from professional phase to personal interests
        # one warm sentence before asking the hobby question
        p2_transition = _claude_text_local(
            client,
            prompt=(
                f"Generate one short, warm sentence to transition into asking {display_name} "
                "about their personal interests or hobbies. "
                "RULES: "
                "- Do NOT mention office, workplace, or 9-to-5. "
                "- Must work for anyone: student, farmer, politician, chef. "
                "- Do not ask the hobby question itself — just set the tone. "
                "- Keep it generic and warm. "
                f"Examples: "
                f"'Now {display_name}, I'd love to know what you enjoy in your personal time.' "
                f"'I'd also love to learn a bit about your interests and passions, {display_name}.' "
                "Keep it natural, 10-18 words. Return only the sentence."
            ),
            fallback=f"I'd also love to learn a bit about your interests and passions, {display_name}.",
        )
        time.sleep(1.0)
        print(f"\n  {p2_transition}")
        speak(p2_transition)
        time.sleep(0.5)

        hobby_q = system.conversation_starter.generate_hobby_discovery_question(
            display_name, client=client
        )
        print(f"\n  {hobby_q}")
        hq_audio, hq_byte = speak_and_record(hobby_q, recording_dir, "hobby_discovery_question")
        timings.append(("hobby_discovery_first_byte", hq_byte))

        print("\n[Please share your interests]")
        hobby_answer, hobby_audio, hobby_end = _listen_with_meta_handling(
            hobby_q, recording_dir, "hobby_discovery", client
        )
        last_response_end = hobby_end
        timings.append(("hobby_response_end", hobby_end))

        if hobby_answer and hobby_answer.lower() == "quit":
            _end_interview(display_name); return

        save_response(
            response_dir,
            {"category_type": "hobby_discovery", "category_name": "Personal Interests",
             "question_text": hobby_q, "willingness_level": str(willingness_level).lower()},
            hobby_answer if hobby_answer else "[No response]",
            recording_dir, "hobby_discovery", hobby_audio, hq_audio,
        )

        # ══════════════════════════════════════════════════════════════════════
        # PHASE 3 — Hobby questions
        # ══════════════════════════════════════════════════════════════════════
        if hobby_answer:
            # small natural pause between Phase 2 answer and hobby questions
            time.sleep(1.5)

            # Claude generates natural transition + clean hobby list
            intro_sentence, hobby_plan = _generate_hobby_intro(
                client, hobby_answer, display_name, num_questions=HOBBY_QUESTION_COUNT
            )

            # speak transition FIRST — then print the phase header
            hi_audio, hi_byte = speak_and_record(intro_sentence, recording_dir, "hobby_intro")
            print("\n=== PHASE 3: Hobby Questions ===")
            print(f"  Transition: {intro_sentence}")
            timings.append(("hobby_intro_first_byte", hi_byte))

            # start background generation for each unique hobby in the plan
            unique_hobbies = list(dict.fromkeys(hobby_plan))  # preserve order, deduplicate
            dataset_events: dict = {}
            for hobby in unique_hobbies:
                system.categories["hobby"] = hobby
                evt = threading.Event()
                dataset_events[hobby] = evt
                threading.Thread(
                    target=generate_dataset_background,
                    args=(system.question_generator, "hobby", hobby, system, evt),
                    daemon=True,
                ).start()

            hobby_last_end: Optional[float] = None

            for i in range(HOBBY_QUESTION_COUNT):
                question_count  += 1
                current_hobby    = hobby_plan[i]
                system.categories["hobby"] = current_hobby

                hq_prep = time.time()
                timings.append((f"hobby_q{i+1}_prep", hq_prep))

                # wait briefly for dataset on first use of each hobby
                if i == 0 or (i > 0 and hobby_plan[i] != hobby_plan[i-1]):
                    evt = dataset_events.get(current_hobby)
                    if evt:
                        evt.wait(timeout=3)
                        if evt.is_set():
                            system.update_available_datasets("hobby")

                question = system.get_question_by_category("hobby", willingness_level)

                print(f"\nQ{question_count} [Hobby {i+1}/{HOBBY_QUESTION_COUNT}] [{current_hobby}]")
                print(f"  {question['question_text']}")

                hq_file, hq_first = speak_and_record(
                    question["question_text"], recording_dir,
                    f"q{question_count}_hobby_question"
                )
                timings.append((f"hobby_q{i+1}_first_byte", hq_first))

                if i > 0 and hobby_last_end:
                    gap = hq_first - hobby_last_end
                    hobby_gaps.append(gap); all_gaps.append(gap)
                    timings.append((f"hobby_q{i+1}_true_gap", gap))
                    print(f"  ⏱ {gap:.2f}s")

                print("\n[Your thoughts]")
                answer, audio_data, proc_end = _listen_with_meta_handling(
                    question["question_text"], recording_dir,
                    f"q{question_count}_hobby", client
                )
                hobby_last_end    = proc_end
                last_response_end = proc_end
                timings.append((f"hobby_q{i+1}_response_end", proc_end))

                if answer and answer.lower() == "quit":
                    _end_interview(display_name); return

                save_response(response_dir, question,
                              answer if answer else "[No response]",
                              recording_dir, f"q{question_count}_hobby",
                              audio_data, hq_file)

                if audio_data is not None and len(audio_data) > 0:
                    willingness_level, _ = system.update_willingness_level(audio_data)

        # ══════════════════════════════════════════════════════════════════════
        # CONCLUSION
        # ══════════════════════════════════════════════════════════════════════
        elapsed = time.time() - start_time
        print(f"\nInterview complete — {int(elapsed//60)}m {int(elapsed%60)}s | "
              f"{question_count} questions")
        _print_timing_stats(all_gaps, prof_gaps, hobby_gaps)
        _write_timing_csv(response_dir, timings, all_gaps, prof_gaps, hobby_gaps)

        ask_feedback_question(display_name, response_dir, recording_dir, client)

        closing = _claude_text_local(
            client,
            prompt=(
                f"Generate a warm 1-2 sentence closing thanking {display_name} "
                "for participating in a personality and interests interview (like a podcast). "
                "This is NOT a job interview — do not mention hiring, teams, joining, or careers. "
                "Just thank them warmly for their time and the conversation. "
                "Professional but personal tone. Return only the sentences."
            ),
            fallback=f"Thank you so much for your time, {display_name}. It has been a genuine pleasure speaking with you.",
        )
        print(f"\n{closing}"); speak(closing)
        time.sleep(1)

        outro = (
            "And thank you to our audience for being here. "
            "From all of us at the RoboJEC team — Dr. Agya Mishra, Devarshi Dixit, and Sanskriti Jain — "
            "have a wonderful day."
        )
        print(outro); speak(outro)

    except KeyboardInterrupt:
        print("\nInterview interrupted.")
        elapsed = time.time() - start_time
        print(f"Completed {question_count} questions in {int(elapsed//60)}m {int(elapsed%60)}s")
        _print_timing_stats(all_gaps, prof_gaps, hobby_gaps)
        _write_timing_csv(response_dir, timings, all_gaps, prof_gaps, hobby_gaps, interrupted=True)


# ── run_interview entry point ──────────────────────────────────────────────────

def run_interview(api_key: Optional[str] = None) -> None:
    key = api_key or ANTHROPIC_API_KEY
    if not key:
        key = input("Enter your Anthropic API key: ").strip()

    client    = Anthropic(api_key=key)
    user_info = get_user_info(client=client)

    if user_info is None:
        print("  [RoboJEC] No activation detected. Goodbye.")
        return

    questions_dir = Path(QUESTIONS_DIR)
    questions_dir.mkdir(parents=True, exist_ok=True)

    prof_cats   = user_info["profession_categories"]
    main_cat    = prof_cats["main_category"]
    subcategory = prof_cats["subcategory"]

    print("\nChecking question datasets…")
    qgen            = PersonalityQuestionsGenerator(client, questions_dir)
    datasets_to_gen = []

    for cat_type, cat_name in [("main", main_cat), ("subcategory", subcategory)]:
        fp = qgen._file_path(cat_type, cat_name)
        if fp.exists():
            print(f"  ✓ {cat_type}: {cat_name}")
        else:
            print(f"  ✗ {cat_type}: {cat_name} — generating in background")
            datasets_to_gen.append((cat_type, cat_name))

    system = PersonalityInterviewSystem(
        client=client,
        questions_dir=questions_dir,
        main_category=main_cat,
        subcategory=subcategory,
        hobby="",
        user_context=user_info.get("context", "professional"),
        user_field=user_info.get("profession_categories", {}).get("field", ""),
    )

    for cat_type, cat_name in datasets_to_gen:
        threading.Thread(
            target=generate_dataset_background,
            args=(qgen, cat_type, cat_name, system),
            daemon=True,
        ).start()

    conduct_interview(
        system,
        user_info["name"],
        user_info["recording_dir"],
        client=client,
        user_info=user_info,
    )


# ── helpers ────────────────────────────────────────────────────────────────────

def _end_interview(display_name: str) -> None:
    msg = f"\nThank you for sharing, {display_name}. It's been a pleasure speaking with you."
    print(msg); speak(msg)