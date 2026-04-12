"""
RoboJEC — Personality Interview System
=======================================
Usage:
    python main.py

RoboJEC listens silently for 5 seconds on startup.
- Nothing heard → exits cleanly
- Something heard → interview begins
- Name mentioned in intro → used directly, no need to ask again
"""

from robojec.pipeline.interview_runner import run_interview

if __name__ == "__main__":
    run_interview()