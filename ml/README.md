# 🤖 Mark XXXIX — ML Intent Classifier (Kaggle Edition)

Machine learning upgrade powered by a **real Kaggle dataset** merged with Mark XXXIX's tool intents.

---

## 📊 Dataset

**Sources combined:**

| Source | Dataset | Intents | Samples |
|--------|---------|---------|---------|
| 🏆 **Kaggle** | [Chatbots Intent Recognition Dataset](https://www.kaggle.com/datasets/elvinagammed/chatbots-intent-recognition-dataset) | 22 | 143 |
| 🤖 **Mark XXXIX** | Custom tool intents | 15 | 185 |
| **Total** | Combined | **37** | **328** |

**CV Accuracy: 74%** across 37 intent classes

---

## 📁 Files

```
ml/
├── kaggle_Intent.json       ← Raw Kaggle dataset (original)
├── intents_dataset.json     ← Combined dataset (Kaggle + Mark XXXIX)
├── intent_model.pkl         ← Trained TF-IDF + LinearSVC model
├── train_intent_model.py    ← Retrain script
└── README.md

actions/
└── intent_classifier.py     ← Jarvis action (plug-and-play)
```

---

## 🧠 Intent Coverage

### Kaggle Conversational Intents (22)
`Greeting`, `GreetingResponse`, `CourtesyGreeting`, `CourtesyGreetingResponse`,
`CurrentHumanQuery`, `NameQuery`, `RealNameQuery`, `TimeQuery`, `Thanks`,
`NotTalking2U`, `UnderstandQuery`, `Shutup`, `Swearing`, `GoodBye`,
`CourtesyGoodBye`, `WhoAmI`, `Clever`, `Gossip`, `Jokes`, `PodBayDoor`,
`PodBayDoorResponse`, `SelfAware`

### Mark XXXIX Tool Intents (15)
`open_app`, `web_search`, `weather_report`, `send_message`, `reminder`,
`youtube_video`, `screen_process`, `computer_settings`, `file_controller`,
`code_helper`, `dev_agent`, `browser_control`, `flight_finder`,
`game_updater`, `shutdown_jarvis`

---

## 🚀 Integration (4 steps)

See `INTEGRATION_GUIDE.py` for exact copy-paste code.

### Retrain anytime:
```bash
python ml/train_intent_model.py
```

### Test the classifier:
```python
from actions.intent_classifier import classify_intent
print(classify_intent("open spotify"))
# → {'intent': 'open_app', 'confidence': 0.59, 'reliable': True, 'tool': 'open_app'}

print(classify_intent("tell me a joke"))
# → {'intent': 'Jokes', 'confidence': 0.48, 'reliable': True, 'tool': 'conversational'}
```
