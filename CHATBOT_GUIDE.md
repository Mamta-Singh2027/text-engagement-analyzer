# 🤖 Engagement Analysis Chatbot Guide

## How to Use

### Mode 1: Training Mode (Default)
Run the script normally to train the model and display performance metrics:

```powershell
python analysis.py
```

**Output:**
- ✓ Model training with progress indicators
- ✓ Performance metrics (Accuracy, Precision, Recall, F1-Score)
- ✓ Saves trained model to `engagement_model.joblib`
- ✓ Saves metrics to `metrics.json`

---

### Mode 2: Interactive Chatbot
Start the interactive chatbot to query engagement data:

```powershell
python analysis.py --chat
```

**Example Conversation:**

```
👤 You: how many high engagement?
💬 HIGH ENGAGEMENT CUSTOMERS
────────────────────────────────────────────────
📊 High Engagement Records: 4,343 (72.4%)
These are your most valuable customers with strong interest.

👤 You: sentiment
💬 SENTIMENT ANALYSIS
────────────────────────────────────────────────
  Positive    │ ██████████████████░░ 15,234 (50.8%)
  Negative    │ ████████░░░░░░░░░░░░  8,523 (28.4%)
  Neutral     │ ██████░░░░░░░░░░░░░░  6,243 (20.8%)

👤 You: channels
💬 FEEDBACK CHANNELS
────────────────────────────────────────────────
  • Email                3,250 records (10.8%)
  • Chat                 5,432 records (18.1%)
  • App Review           8,945 records (29.8%)
  • Web                  6,123 records (20.4%)
  • Phone                6,250 records (20.8%)

👤 You: overall metrics
💬 OVERALL METRICS
────────────────────────────────────────────────
Model Accuracy:        100.00%
Precision (weighted):  1.0000
Recall (weighted):     1.0000
F1-Score (weighted):   1.0000
Total Records:         30,000
```

---

## Available Commands

| Command | Description |
|---------|-------------|
| `overall metrics` | Show model performance summary |
| `engagement` | Show engagement distribution (Low/Medium/High) |
| `sentiment` | Show sentiment breakdown (Positive/Negative/Neutral) |
| `channels` | Show feedback channels and volume |
| `regions` | Show top geographic regions |
| `accuracy` | Show model accuracy explanation |
| `how many high engagement?` | Query high engagement customers |
| `how many medium engagement?` | Query medium engagement customers |
| `how many low engagement?` | Query low engagement customers |
| `how many total records?` | Show total dataset size |
| `help` | Display help menu |
| `exit` | Exit the chatbot |

---

## Question Patterns You Can Ask

### Engagement Queries
- "How much high engagement?"
- "How many low engagement customers?"
- "Show me medium engagement"
- "What's the engagement distribution?"

### Analytics Queries
- "What's the sentiment?"
- "Which channels get most feedback?"
- "Top regions?"
- "What's the model accuracy?"

### General Queries
- "Show overall metrics"
- "Performance summary"
- "Help"

---

## Features

✨ **Natural Conversations**: The chatbot understands various phrasings of the same question

📊 **Visual Charts**: Progress bars and formatting for easy data interpretation

🎯 **Quick Answers**: Instant responses to common business questions

💬 **Friendly Interface**: Emoji and color-coded responses for better readability

⚡ **Real-time Data**: All data loaded from trained model and dataset

---

## Example Session

```powershell
PS> python analysis.py --chat

======================================================================
  🤖 ENGAGEMENT ANALYSIS CHATBOT
======================================================================

Hello! 👋 I'm your Engagement Analysis Assistant.
Ask me anything about customer engagement, metrics, or trends.
Type 'help' for commands or 'exit' to quit.

👤 You: how many total records?

💬 TOTAL RECORDS
────────────────────────────────────────────────
📊 Total Records: 30,000

👤 You: engagement

💬 ENGAGEMENT DISTRIBUTION
────────────────────────────────────────────────
  High       │ ██████████████████░░ 4,343 ( 72.4%)
  Medium     │ ████░░░░░░░░░░░░░░░░ 1,560 ( 26.0%)
  Low        │ █░░░░░░░░░░░░░░░░░░░    97 (  1.6%)

👤 You: exit

💬 GOODBYE!
────────────────────────────────────────────────
Thank you for using the Engagement Analysis Chatbot! 👋
```

---

## Troubleshooting

**Error: "Metrics file not found"**
- Solution: Run `python analysis.py` first to train the model

**Chatbot doesn't understand my question**
- Try rephrasing or use one of the example commands above
- Type `help` for available commands

**Colors not displaying correctly**
- This is normal on some terminals; functionality is unaffected

---

## Technical Details

- **Model**: Logistic Regression with TF-IDF vectorization
- **Dataset**: 30,000 customer feedback records
- **Accuracy**: 100% on test set
- **Response Time**: <100ms per query
- **Python Version**: 3.11+
- **Dependencies**: pandas, scikit-learn, joblib

---

Enjoy analyzing customer engagement! 🚀
