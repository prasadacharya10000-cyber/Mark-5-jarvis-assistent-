# ─────────────────────────────────────────────────────────────────────────────
# HOW TO INTEGRATE THE ML INTENT CLASSIFIER INTO main.py
# ─────────────────────────────────────────────────────────────────────────────
#
# Step 1 ── Add this import near the top of main.py, after the other action imports:
#
#     from actions.intent_classifier import intent_classifier, classify_intent, suggest_tool
#
#
# Step 2 ── Add this tool declaration to the TOOL_DECLARATIONS list in main.py:
#
INTENT_TOOL_DECLARATION = {
    "name": "intent_classifier",
    "description": (
        "Uses a local ML model (trained on a public chatbot intents dataset) "
        "to classify the intent behind a user utterance. "
        "Use this when you want to double-check which tool to call, "
        "when the user asks 'what did I mean by that', "
        "or to show intent confidence scores. "
        "Also useful when routing is ambiguous."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "text": {
                "type": "STRING",
                "description": "The user utterance to classify"
            },
            "threshold": {
                "type": "NUMBER",
                "description": "Confidence threshold 0-1 (default: 0.35)"
            },
            "verbose": {
                "type": "BOOLEAN",
                "description": "Include top-5 scores in output (default: false)"
            }
        },
        "required": ["text"]
    }
}

#
# Step 3 ── Add this elif branch in the _execute_tool method, alongside the other tools:
#
#     elif name == "intent_classifier":
#         r = await loop.run_in_executor(
#             None,
#             lambda: intent_classifier(parameters=args, player=self.ui, speak=self.speak)
#         )
#         result = r or "Intent classified."
#
#
# Step 4 ── (Optional) Add smart pre-classification before Gemini responds.
#           In the _receive_audio method, after you get the user's full transcript,
#           you can call suggest_tool(full_in) to log the ML prediction:
#
#     if full_in:
#         self.ui.write_log(f"You: {full_in}")
#         ml_tool = suggest_tool(full_in)
#         if ml_tool:
#             self.ui.write_log(f"[ML Hint] Suggested tool: {ml_tool}")
#
# ─────────────────────────────────────────────────────────────────────────────
