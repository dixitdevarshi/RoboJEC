import time
from typing import Any, Dict, Optional

from anthropic import Anthropic

from robojec.core.samvad import SamvadGenerator
from robojec.utils.audio import (
    create_user_recording_directory,
    listen_and_save_name,
)
from robojec.utils.profession import (
    build_display_name,
    generate_specialisation_examples,
    recognize_profession,
)
from robojec.utils.text_utils import extract_number_from_text
from robojec.utils.tts import speak


# ── single Claude text helper ──────────────────────────────────────────────────

def _claude_text(
    client: Optional[Anthropic],
    prompt: str,
    max_tokens: int = 80,
    fallback: str = "",
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
    except Exception as exc:
        print(f"  [Claude] Call failed: {exc}")
        return fallback


# ── opening sequence generators ───────────────────────────────────────────────

def _generate_opening(client: Optional[Anthropic]) -> str:
    return _claude_text(
        client,
        prompt=(
            "You are RoboJEC, an AI interview system. "
            "Generate a warm, professional 2-3 sentence opening to start a personality interview. "
            "Introduce yourself as RoboJEC briefly, welcome the guest, "
            "and end by asking for their name. "
            "Tone: warm and professional — not casual, not theatrical. "
            "Do NOT use 'What should I call you' or 'Mind sharing your name'. "
            "End with something natural like 'May I have your name?' "
            "Return only the statement, nothing else."
        ),
        max_tokens=150,
        fallback=(
            "Hello, welcome to RoboJEC. I'm an AI system designed for meaningful conversations. "
            "It's wonderful to have you here today. Before we begin, may I have your name?"
        ),
    )


def _generate_name_retry(client: Optional[Anthropic]) -> str:
    return _claude_text(
        client,
        prompt=(
            "Generate one short polite sentence asking someone to repeat their name. "
            "Professional tone, not casual, max 12 words. Return only the sentence."
        ),
        fallback="I apologise, I didn't quite catch that. Could you please tell me your name?",
    )


def _generate_acknowledgement(client: Optional[Anthropic], name: str) -> str:
    return _claude_text(
        client,
        prompt=(
            f"Generate one short warm acknowledgement of someone's name: {name}. "
            "One sentence, professional, not over the top. "
            f"Example: 'It's a pleasure to meet you, {name}.' "
            "Return only the sentence."
        ),
        fallback=f"It's a pleasure to meet you, {name}.",
    )


def _generate_profession_question(client: Optional[Anthropic], display_name: str) -> str:
    return _claude_text(
        client,
        prompt=(
            f"Generate one warm, simple question asking {display_name} what they do or what their background is. "
            "STRICT RULES: "
            "- Do NOT ask what brings them joy, fulfillment, or passion. "
            "- Do NOT ask what keeps them busy or what drives them. "
            "- Do NOT assume office, workplace, research, or any context. "
            "- Ask ONLY what they do — their job, profession, or field of study. "
            "- Must work for anyone: student, doctor, farmer, politician, chef, engineer. "
            "- Warm and curious tone. 8-15 words. Address them by name. "
            "Good examples: "
            f"'{display_name}, what do you do — what is your profession or field?' "
            f"'{display_name}, I'd love to know what you do professionally.' "
            f"'{display_name}, tell me about your background — what field are you in?' "
            "Return only the question, nothing else."
        ),
        fallback=f"{display_name}, I'd love to know about your background — what do you do?",
    )


def _generate_years_question(
    display_name: str,
    profession_categories: dict,
) -> str:
    """
    Build a years question directly from what we know — no Claude call needed.
    We have context, field, role — enough to write a clean question.
    """
    context      = profession_categories.get("context", "professional")
    years_context = profession_categories.get("years_context", "working")
    field        = profession_categories.get("field", "").strip()
    role         = profession_categories.get("role", "").strip()

    # pick subject for the question:
    # - use field if specific (e.g. "Cardiology", "Machine Learning")
    # - use role only if it's a self-complete profession (actor, politician, chef...)
    # - otherwise no subject → generic question
    _SKIP_SUBJECTS = {
        "student", "professional", "worker", "person", "other", "",
        "engineer", "doctor", "developer", "programmer", "scientist",
        "researcher", "manager", "consultant", "analyst", "designer",
        "specialist", "technician", "lecturer", "professor",
    }
    if field and field.lower() not in ["other", "your field", "[your field]", ""]:
        subject = field
    elif (role and role.lower() not in _SKIP_SUBJECTS
          and context != "student"):
        subject = role.lower()
    else:
        subject = None

    if context == "student":
        if subject:
            return f"How long have you been studying {subject}, {display_name}?"
        return f"How long have you been studying, {display_name}?"

    elif years_context == "practising":
        if subject:
            return f"How long have you been practising {subject}, {display_name}?"
        return f"How long have you been in your field, {display_name}?"

    elif years_context == "researching":
        if subject:
            return f"How long have you been researching {subject}, {display_name}?"
        return f"How many years have you been in research, {display_name}?"

    else:
        # working / professional
        if subject:
            _AS_A_ROLES = {
                "actor", "actress", "chef", "pilot", "journalist", "politician",
                "musician", "singer", "dancer", "photographer", "filmmaker",
                "director", "architect", "accountant", "dentist", "pharmacist",
                "veterinarian", "firefighter", "soldier", "farmer", "carpenter",
                "electrician", "plumber", "librarian", "economist",
            }
            if subject.lower() in _AS_A_ROLES:
                article = "an" if subject.lower()[0] in "aeiou" else "a"
                return f"How long have you been working as {article} {subject.lower()}, {display_name}?"
            return f"How many years have you been working in {subject}, {display_name}?"
        return f"How many years of experience do you have, {display_name}?"


# ── wake window ────────────────────────────────────────────────────────────────

def _wake_window(starter: SamvadGenerator) -> Optional[str]:
    """
    Listen silently for 5 seconds.
    Returns:
        None   -> nothing heard or noise/single word -> exit
        ""     -> real speech heard but no name found
        <name> -> name extracted from intro
    """
    print("  [Wake] Listening for activation... (5 seconds)")
    text, _ = listen_and_save_name(
        None, "wake",
        silence_threshold=5,
        min_speech_duration=2.0,   # raised from 0.5 -- ignore short noise bursts
        max_recording_duration=5,
    )

    if not text or not text.strip():
        return None

    # reject single-word captures -- likely noise or ambient sound
    words = text.strip().split()
    if len(words) < 2:
        print(f"  [Wake] Ignoring noise: '{text}'")
        return None

    name = starter.extract_name(text)
    return name if (name and name != "Friend") else ""
    return name if (name and name != "Friend") else ""


# ── main function ──────────────────────────────────────────────────────────────

def get_user_info(client: Optional[Anthropic] = None) -> Optional[Dict[str, Any]]:
    """
    Returns None if nothing heard in wake window.
    Returns structured user info dict otherwise.
    """
    print("\n  [RoboJEC] Starting — listening for activation…")
    starter = SamvadGenerator()

    # ── wake window ───────────────────────────────────────────────────────────
    wake_result = _wake_window(starter)
    if wake_result is None:
        print("  [Wake] Nothing heard. Exiting.")
        return None

    name_from_intro = wake_result if wake_result != "" else None

    print("\n--- RoboJEC Interview System ---\n")

    # ── name acquisition ──────────────────────────────────────────────────────
    if name_from_intro:
        name = name_from_intro
        ack  = _generate_acknowledgement(client, name)
        print(ack); speak(ack)
        time.sleep(0.5)
    else:
        opening = _generate_opening(client)
        print(opening); speak(opening)
        time.sleep(0.5)

        name_text, _ = listen_and_save_name(None, "name_initial")
        name          = starter.extract_name(name_text) if name_text else None

        for retry in range(3):
            if name and name != "Friend":
                break
            retry_q = _generate_name_retry(client)
            print(retry_q); speak(retry_q)
            name_text, _ = listen_and_save_name(None, f"name_retry_{retry}")
            name = starter.extract_name(name_text) if name_text else None

        if not name or name == "Friend":
            name = "Friend"

        ack = _generate_acknowledgement(client, name)
        print(ack); speak(ack)
        time.sleep(0.5)

    # ── consent disclaimer ───────────────────────────────────────────────────
    disclaimer = (
        "Before we begin, I'd like to let you know that this conversation "
        "will be recorded for research and training purposes. "
        "By continuing, you consent to this recording."
    )
    print(disclaimer)
    speak(disclaimer)
    time.sleep(3)

    # ── recording directory ───────────────────────────────────────────────────
    recording_dir = create_user_recording_directory(name)

    # ── profession — first pass ───────────────────────────────────────────────
    # Use plain name for now; display_name set after we know title
    prof_q = _generate_profession_question(client, name)
    print(f"\n{prof_q}"); speak(prof_q)

    prof_text, _ = listen_and_save_name(recording_dir, "profession_initial")
    profession_text = prof_text.strip() if prof_text else ""

    if not profession_text:
        retry_msg = f"I didn't quite catch that, {name}. Could you tell me about your field?"
        print(retry_msg); speak(retry_msg)
        prof_text, _ = listen_and_save_name(recording_dir, "profession_retry")
        profession_text = prof_text.strip() if prof_text else "unknown"

    profession_categories = recognize_profession(
        profession_text, years_experience=None, client=client,
    )

    # ── clarification loop ────────────────────────────────────────────────────
    clarify_attempts = 0
    while not profession_categories.get("is_complete", True) and clarify_attempts < 2:
        clarification_q = profession_categories.get("needs_clarification", "")
        if not clarification_q:
            break

        print(f"\n{clarification_q}"); speak(clarification_q)
        clarify_text, _ = listen_and_save_name(recording_dir, f"clarify_{clarify_attempts}")
        clarify_attempts += 1

        if not clarify_text or len(clarify_text.split()) < 2:
            # person couldn't answer — give examples
            role = profession_categories.get("role", profession_text)
            examples = generate_specialisation_examples(client, role)
            if examples:
                print(f"\n{examples}"); speak(examples)
                # ask again
                clarify_text, _ = listen_and_save_name(
                    recording_dir, f"clarify_{clarify_attempts}_after_examples",
                )

        if clarify_text:
            profession_text       = f"{profession_text}. {clarify_text}"
            profession_categories = recognize_profession(
                profession_text, years_experience=None, client=client,
            )

    # ── build display name with title ─────────────────────────────────────────
    display_name = build_display_name(name, profession_categories)
    if display_name != name:
        print(f"  [Title] Using display name: {display_name}")

    # ── years question ────────────────────────────────────────────────────────
    years_q = _generate_years_question(display_name, profession_categories)

    print(f"\n{years_q}"); speak(years_q)

    years_experience = None
    attempt = 1
    while years_experience is None:
        exp_text, _ = listen_and_save_name(recording_dir, f"experience_{attempt}")
        years_experience = extract_number_from_text(exp_text)

        if years_experience is not None and years_experience >= 0:
            break
        elif years_experience is not None and years_experience < 0:
            msg = f"That doesn't seem right — could you give me a positive number, {display_name}?"
            print(msg); speak(msg)
            years_experience = None
        else:
            msg = "Could you give me a number — like 2 years or 5 years?"
            print(msg); speak(msg)
            attempt += 1

        if attempt > 3:
            years_experience = 0
            break

    # ── final recognition with years ─────────────────────────────────────────
    profession_categories = recognize_profession(
        profession_text, years_experience=years_experience, client=client,
    )
    display_name = build_display_name(name, profession_categories)

    context = profession_categories.get("context", "professional")
    print(f"\n  Name         : {name}")
    print(f"  Display name : {display_name}")
    print(f"  Field        : {profession_categories.get('field', '')}")
    print(f"  Role         : {profession_categories.get('subcategory', '')}")
    print(f"  Context      : {context}")
    print(f"  Experience   : {years_experience} years ({profession_categories.get('seniority', '')})")
    print(f"  Source       : {profession_categories.get('_source', '')}")
    print("\nPreparing your personalised interview…\n")

    return {
        "name":                  name,
        "display_name":          display_name,
        "profession_text":       profession_text,
        "profession_categories": profession_categories,
        "years_experience":      years_experience,
        "context":               context,
        "hobbies":               [],
        "recording_dir":         recording_dir,
    }