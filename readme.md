# Mark XXXIX — AI Desktop Assistant

A modular, Iron Man-inspired AI desktop assistant with ML-based intent classification, browser automation, file management, and a suite of productivity actions.

## Features
- ML intent classifier trained on Kaggle dataset for natural language understanding
- Browser control and automation
- File processing and file controller
- Flight finder
- YouTube video search and automation
- Weather reports
- Reminders and task management
- Code helper and dev agent
- Desktop and computer control
- WhatsApp/message sending
- Screen processor

## Tech Stack
Python · LLMs · NLP · scikit-learn · PyAutoGUI · Selenium · Transformers

## Project Structure
```
mark-xxxix/
├── main.py              ← Entry point
├── ui.py                ← Desktop UI
├── actions/             ← All action modules
│   ├── browser_control.py
│   ├── file_processor.py
│   ├── flight_finder.py
│   ├── intent_classifier.py
│   └── ...
├── agent/               ← Planner, executor, task queue
├── ml/                  ← Intent model and training data
├── memory/              ← Memory and config manager
└── core/                ← Prompt config
```

## Getting Started

### Installation
```bash
git clone https://github.com/prasadacharya10000-cyber/mark-xxxix-ai-assistant.git
cd mark-xxxix-ai-assistant
pip install -r requirements.txt
python setup.py
```

### Run
```bash
python main.py
```

## Intent Classification
The assistant uses a Kaggle-trained ML model (`ml/intent_model.pkl`) to classify user commands into actions. To retrain:
```bash
python ml/train_intent_model.py
```
