"""
RoboJEC Web Interface
=====================
Run with:  python app.py
Then open: http://localhost:5000
"""

import os
import threading
from pathlib import Path

from flask import Flask, render_template, send_file, jsonify
from flask_socketio import SocketIO, emit

import robojec.utils.tts as _tts_module

app      = Flask(__name__)
app.config["SECRET_KEY"] = "robojec-ui-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── global interview state ─────────────────────────────────────────────────────
_interview_thread: threading.Thread | None = None
_interview_running                         = False
_stop_event                                = threading.Event()
_last_session_name: str                    = ""   # track latest guest name for CSV download
_current_session_csv: str                  = ""   # path to current session's CSV — only this is downloadable


# ── event emitter (thread-safe) ────────────────────────────────────────────────

def _emit(event: str, data: dict) -> None:
    socketio.emit(event, data)


# ── patch speak ────────────────────────────────────────────────────────────────

_original_speak            = _tts_module.speak
_original_speak_and_record = _tts_module.speak_and_record


def _patched_speak(text: str, rate: int = _tts_module.TTS_RATE) -> None:
    if _stop_event.is_set():
        return
    _emit("system_speech", {"text": text, "type": "speak"})
    _original_speak(text, rate)


def _patched_speak_and_record(text, recording_dir, recording_id, rate=_tts_module.TTS_RATE):
    if _stop_event.is_set():
        return None, 0.0
    _emit("system_speech", {"text": text, "type": "question"})
    return _original_speak_and_record(text, recording_dir, recording_id, rate)


_tts_module.speak            = _patched_speak
_tts_module.speak_and_record = _patched_speak_and_record

import robojec.pipeline.interview_runner as _runner
import robojec.pipeline.user_info       as _user_info_module

# patch save_response to capture the session CSV path
_original_save_response = _runner.save_response

def _patched_save_response(response_dir, question, answer, recording_dir,
                            recording_id, audio_data=None, question_audio_file=None):
    global _current_session_csv
    result = _original_save_response(
        response_dir, question, answer, recording_dir,
        recording_id, audio_data, question_audio_file
    )
    csv_path = response_dir / "interview_responses.csv"
    if csv_path.exists():
        _current_session_csv = str(csv_path)
    return result

_runner.save_response = _patched_save_response

_runner.speak            = _patched_speak
_runner.speak_and_record = _patched_speak_and_record
_user_info_module.speak  = _patched_speak


# ── patch listen to emit state + check stop ────────────────────────────────────

import robojec.utils.audio as _audio_module

_original_listen_and_save      = _audio_module.listen_and_save
_original_listen_and_save_name = _audio_module.listen_and_save_name


def _patched_listen_and_save(recording_dir, question_id, **kwargs):
    if _stop_event.is_set():
        import numpy as np
        return "quit", np.array([], dtype=float), 0.0
    _emit("listening_state", {"state": "listening"})
    result = _original_listen_and_save(recording_dir, question_id, **kwargs)
    text   = result[0] if result else ""
    if text and not hasattr(text, "kind"):
        _emit("user_speech", {"text": str(text)})
    _emit("listening_state", {"state": "processing"})
    return result


def _patched_listen_and_save_name(recording_dir, question_id, **kwargs):
    if _stop_event.is_set():
        import numpy as np
        return "quit", np.array([], dtype=float)
    _emit("listening_state", {"state": "listening"})
    result = _original_listen_and_save_name(recording_dir, question_id, **kwargs)
    text   = result[0] if result else ""
    if text and isinstance(text, str) and text not in ("quit", ""):
        _emit("user_speech", {"text": text})
    _emit("listening_state", {"state": "idle"})
    return result


_audio_module.listen_and_save       = _patched_listen_and_save
_audio_module.listen_and_save_name  = _patched_listen_and_save_name
_runner.listen_and_save             = _patched_listen_and_save
_runner.listen_and_save_name        = _patched_listen_and_save_name
_user_info_module.listen_and_save_name = _patched_listen_and_save_name


# ── patch willingness ──────────────────────────────────────────────────────────

from robojec.core.interview_system import PersonalityInterviewSystem as _PIS
_original_update_willingness = _PIS.update_willingness_level


def _patched_update_willingness(self, audio_data):
    level, score = _original_update_willingness(self, audio_data)
    _emit("willingness_update", {"level": level.value, "score": round(score, 1)})
    return level, score


_PIS.update_willingness_level = _patched_update_willingness


# ── patch user_info to emit profile ───────────────────────────────────────────

_original_get_user_info = _user_info_module.get_user_info


def _patched_get_user_info(client=None):
    global _last_session_name
    result = _original_get_user_info(client=client)
    if result:
        prof = result.get("profession_categories", {})
        name = result.get("display_name", result.get("name", ""))
        _last_session_name = result.get("name", "")
        _emit("user_profile", {
            "name":      name,
            "field":     prof.get("field", ""),
            "role":      prof.get("subcategory", ""),
            "context":   result.get("context", "professional"),
            "seniority": prof.get("seniority", ""),
            "years":     result.get("years_experience", 0),
        })
    return result


_user_info_module.get_user_info = _patched_get_user_info
_runner.get_user_info           = _patched_get_user_info


# ── interview runner thread ────────────────────────────────────────────────────

def _run_interview_thread():
    global _interview_running
    try:
        _emit("interview_status", {"status": "started"})
        _runner.run_interview()
    except (SystemExit, KeyboardInterrupt):
        # clean exit triggered by end_interview button
        pass
    except Exception as exc:
        if not _stop_event.is_set():
            _emit("interview_error", {"error": str(exc)})
    finally:
        _interview_running = False
        _emit("interview_status", {"status": "ended"})


# ── routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/download-csv")
def download_csv():
    """Download the current session's responses CSV only."""
    if not _current_session_csv:
        return jsonify({
            "error": "No responses recorded in this session yet."
        }), 404

    csv_path = Path(_current_session_csv)
    if not csv_path.exists():
        return jsonify({
            "error": "Session file not found."
        }), 404

    return send_file(
        csv_path,
        as_attachment=True,
        download_name=f"robojec_{_last_session_name or 'session'}_responses.csv",
    )


# ── socket events ──────────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    emit("connection_ok", {"message": "Connected"})


@socketio.on("start_interview")
def on_start_interview():
    global _interview_thread, _interview_running

    if _interview_running:
        emit("interview_error", {"error": "Interview already running"})
        return

    _stop_event.clear()
    _interview_running  = True
    _current_session_csv = ""
    _interview_thread  = threading.Thread(target=_run_interview_thread, daemon=True)
    _interview_thread.start()


@socketio.on("end_interview")
def on_end_interview():
    global _interview_running
    _stop_event.set()
    _interview_running = False

    # Force-interrupt the background thread by raising SystemExit inside it.
    # This works even when the thread is blocked inside a recording call.
    if _interview_thread and _interview_thread.is_alive():
        import ctypes
        tid = _interview_thread.ident
        if tid:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(tid),
                ctypes.py_object(SystemExit),
            )

    _emit("interview_status", {"status": "ended"})


# ── run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("RoboJEC Web Interface")
    print("Open http://localhost:5000")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)