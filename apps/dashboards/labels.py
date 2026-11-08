"""Words, icons and colours the apps show. Plain language first, model jargon never.

Colours were run through the dataviz validator on a white surface: emotions and moods each pass
the adjacent-pair colour-blind and normal-vision checks. Blue and orange sit below 3:1 contrast on
white, so every chart that uses them also prints direct labels and offers a table view.
"""
from __future__ import annotations

import re

# ---- Duolingo-flavoured UI tokens
GREEN, GREEN_D = "#58CC02", "#58A700"
BLUE, BLUE_D = "#1CB0F6", "#1899D6"
RED, RED_D = "#FF4B4B", "#EA2B2B"
YELLOW, YELLOW_D = "#FFC800", "#E5B400"
ORANGE, ORANGE_D = "#FF9600", "#E58700"
PURPLE, PURPLE_D = "#CE82FF", "#A560E8"
INK, INK_2, INK_3 = "#3C3C3C", "#777777", "#AFAFAF"
LINE, SOFT = "#E5E5E5", "#F7F7F7"

EMOTIONS = ["anger", "fear", "joy", "sadness", "surprise"]
EMO_COLOR = {"anger": "#E5484D", "fear": "#A560E8", "joy": "#46A302", "sadness": "#1CB0F6", "surprise": "#FF9600"}
EMO_NAME = {"anger": "Anger", "fear": "Worry", "joy": "Joy", "sadness": "Sadness", "surprise": "Surprise"}
EMO_ICON = {"anger": "😠", "fear": "😟", "joy": "😄", "sadness": "😢", "surprise": "😮"}
NEGATIVE = ["anger", "fear", "sadness"]

# order matters: neighbours are the pairs the validator checked
STATES = ["outrage", "disappointment", "frustration", "stress", "delight", "mixed"]
STATE_COLOR = {"outrage": "#E5484D", "disappointment": "#1CB0F6", "frustration": "#FF9600",
               "stress": "#A560E8", "delight": "#46A302", "mixed": "#E0559C"}
STATE_NAME = {"outrage": "Outraged", "disappointment": "Disappointed", "frustration": "Frustrated",
              "stress": "Stressed", "delight": "Delighted", "mixed": "Mixed feelings"}
STATE_ICON = {"outrage": "🤬", "disappointment": "😞", "frustration": "😤", "stress": "😰", "delight": "🥳",
              "mixed": "🤔"}
STATE_HELP = {
    "outrage": "Very angry and intense. Needs a human quickly.",
    "disappointment": "Let down, but not angry.",
    "frustration": "Angry and let down at the same time.",
    "stress": "Worried or anxious.",
    "delight": "Happy and pleasantly surprised.",
    "mixed": "Praise and complaint in the same message.",
}
# priority when one headline mood is needed for a message
STATE_PRIORITY = ["outrage", "frustration", "stress", "disappointment", "mixed", "delight"]

TONE = {
    "frustration": "Acknowledge the repeated effort, then give a concrete next step and a time.",
    "outrage": "Apologise without excuses and offer escalation to a supervisor.",
    "stress": "Reassure first, then explain what is already secured (account, payment).",
    "disappointment": "Empathise and offer a specific remedy.",
    "delight": "Thank them and invite a review.",
}

ASPECT_NAME = {
    "app_site": "App & website", "delivery": "Delivery", "product": "Product", "refund": "Refunds & billing",
    "staff": "Staff & service", "other": "Everything else",
    "assignments": "Assignments", "content": "Course content", "exams": "Exams & grading",
    "teaching": "Teaching", "workload": "Difficulty & workload",
}
ASPECT_ICON = {"app_site": "📱", "delivery": "📦", "product": "🛍️", "refund": "💳", "staff": "🎧", "other": "💬",
               "assignments": "📝", "content": "📚", "exams": "🎯", "teaching": "🧑‍🏫", "workload": "⏳"}
# topic lines in the trend chart: five validated categorical slots
ASPECT_COLORS = ["#2A78D6", "#EB6834", "#1BAF7A", "#E87BA4", "#6250D6", "#8A8A8A"]

LANG_NAME = {"eng": "English", "esp": "Spanish", "ptbr": "Portuguese", "deu": "German", "fra": "French",
             "ind": "Indonesian", "hin": "Hindi", "chn": "Chinese / Japanese", "rus": "Russian", "ara": "Arabic",
             "swa": "Swahili", "swe": "Swedish", "mar": "Marathi", "arq": "Arabic (Algeria)", "<unk>": "Not sure"}


def aspect_name(a: str) -> str:
    return ASPECT_NAME.get(a, a.replace("_", " ").title())


def aspect_icon(a: str) -> str:
    return ASPECT_ICON.get(a, "🏷️")


def pretty_group(g: str) -> str:
    """course_1372 -> Course 1372, teacher_89 -> Teacher 89, university_reviews -> University (overall)."""
    m = re.fullmatch(r"(course|teacher)_(\d+)", str(g))
    if m:
        return f"{m[1].title()} {m[2]}"
    if g == "university_reviews":
        return "University (overall)"
    return str(g)


def pct(x: float, d: int = 0) -> str:
    return f"{x * 100:.{d}f}%"


def mood_word(v: float) -> str:
    """valence -1..1 to words"""
    if v >= 0.45:
        return "Very positive"
    if v >= 0.15:
        return "Positive"
    if v > -0.15:
        return "Neutral"
    if v > -0.45:
        return "Negative"
    return "Very negative"
