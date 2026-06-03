"""
intent_classifier.py — ML-powered intent classification for Mark XXXIX
Uses a TF-IDF + LinearSVC model trained on:
  - Kaggle: Chatbots Intent Recognition Dataset (22 conversational intents)
  - Mark XXXIX custom dataset (15 tool/action intents)
  Combined: 37 intents, 328 training samples
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
MODEL_PATH   = BASE_DIR / "ml" / "intent_model.pkl"
DATASET_PATH = BASE_DIR / "ml" / "intents_dataset.json"

# ── Singleton model cache ────────────────────────────────────────────────────
_model = None

def _load_model():
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Intent model not found at {MODEL_PATH}. "
            "Run: python ml/train_intent_model.py"
        )
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    logger.info("[IntentClassifier] Model loaded from %s", MODEL_PATH)
    return _model


# ── Intent → Tool/Action mapping ────────────────────────────────────────────
# Mark XXXIX tool intents
INTENT_TO_TOOL = {
    "open_app":          "open_app",
    "web_search":        "web_search",
    "weather_report":    "weather_report",
    "send_message":      "send_message",
    "reminder":          "reminder",
    "youtube_video":     "youtube_video",
    "screen_process":    "screen_process",
    "computer_settings": "computer_settings",
    "file_controller":   "file_controller",
    "code_helper":       "code_helper",
    "dev_agent":         "dev_agent",
    "browser_control":   "browser_control",
    "flight_finder":     "flight_finder",
    "game_updater":      "game_updater",
    "shutdown_jarvis":   "shutdown_jarvis",
    # Kaggle conversational intents → mapped to appropriate responses
    "Greeting":                  "conversational",
    "GreetingResponse":          "conversational",
    "CourtesyGreeting":          "conversational",
    "CourtesyGreetingResponse":  "conversational",
    "CurrentHumanQuery":         "conversational",
    "NameQuery":                 "conversational",
    "RealNameQuery":             "conversational",
    "TimeQuery":                 "computer_settings",   # ask system time
    "Thanks":                    "conversational",
    "NotTalking2U":              "conversational",
    "UnderstandQuery":           "conversational",
    "Shutup":                    "conversational",
    "Swearing":                  "conversational",
    "GoodBye":                   "shutdown_jarvis",
    "CourtesyGoodBye":           "shutdown_jarvis",
    "WhoAmI":                    "conversational",
    "Clever":                    "conversational",
    "Gossip":                    "conversational",
    "Jokes":                     "conversational",
    "PodBayDoor":                "conversational",
    "PodBayDoorResponse":        "conversational",
    "SelfAware":                 "conversational",
}

# ── Core classification ──────────────────────────────────────────────────────
def classify_intent(text: str, confidence_threshold: float = 0.30) -> dict:
    """
    Classify the intent of a user utterance.

    Returns:
        {
            "intent":     str   — predicted intent tag,
            "confidence": float — probability 0-1,
            "reliable":   bool  — True if confidence >= threshold,
            "tool":       str   — suggested Jarvis tool,
            "all_scores": dict  — top-5 {intent: score}
        }
    """
    model      = _load_model()
    text_clean = text.lower().strip()

    proba   = model.predict_proba([text_clean])[0]
    classes = model.classes_
    top_idx = proba.argmax()

    # Top 5 scores
    scored   = sorted(zip(classes, proba), key=lambda x: -x[1])[:5]
    top5     = {cls: round(float(p), 4) for cls, p in scored}

    top_intent = classes[top_idx]
    confidence = float(proba[top_idx])
    tool       = INTENT_TO_TOOL.get(top_intent, "unknown")

    return {
        "intent":     top_intent,
        "confidence": round(confidence, 4),
        "reliable":   confidence >= confidence_threshold,
        "tool":       tool,
        "all_scores": top5,
    }


def suggest_tool(text: str) -> Optional[str]:
    """Return suggested tool name for a user utterance, or None if uncertain."""
    result = classify_intent(text)
    if result["reliable"] and result["tool"] not in ("conversational", "unknown"):
        return result["tool"]
    return None


# ── Jarvis action entry point ────────────────────────────────────────────────
def intent_classifier(parameters: dict, player=None, speak=None) -> str:
    """
    Jarvis action: classify the intent of a user command using the ML model.

    Parameters:
        text      (str)   — utterance to classify
        threshold (float) — confidence threshold (default 0.30)
        verbose   (bool)  — include top-5 scores (default False)
    """
    text      = parameters.get("text", "").strip()
    threshold = float(parameters.get("threshold", 0.30))
    verbose   = bool(parameters.get("verbose", False))

    if not text:
        return "No text provided for intent classification."

    try:
        result = classify_intent(text, confidence_threshold=threshold)
    except FileNotFoundError as e:
        return f"Intent model not available: {e}"
    except Exception as e:
        logger.exception("Intent classification failed")
        return f"Classification error: {e}"

    intent     = result["intent"]
    confidence = result["confidence"]
    reliable   = result["reliable"]
    tool       = result["tool"]

    summary = (
        f"Intent: {intent} | Confidence: {confidence:.0%} | "
        f"Reliable: {'Yes' if reliable else 'No'} | Suggested tool: {tool}"
    )

    if verbose:
        scores_str = ", ".join(f"{k}: {v:.0%}" for k, v in result["all_scores"].items())
        summary += f" | Top scores: [{scores_str}]"

    if player:
        player.write_log(f"[ML] {summary}")

    return summary


# ── Dataset stats ────────────────────────────────────────────────────────────
def get_dataset_stats() -> dict:
    if not DATASET_PATH.exists():
        return {"error": "Dataset not found"}
    with open(DATASET_PATH) as f:
        data = json.load(f)
    intents = data.get("intents", [])
    total   = sum(len(i.get("patterns", i.get("text", []))) for i in intents)
    kaggle  = [i for i in intents if i.get("source") == "kaggle"]
    mark39  = [i for i in intents if i.get("source") == "mark39"]
    return {
        "num_intents":    len(intents),
        "total_samples":  total,
        "kaggle_intents": len(kaggle),
        "mark39_intents": len(mark39),
    }
