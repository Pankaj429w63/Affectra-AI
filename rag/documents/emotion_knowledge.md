# Project Knowledge Document: Affectra AI Emotion Knowledge Base

## SECTION 1 — AFFECTRA AI
Affectra AI is a multimodal emotion and sentiment detection system. The system uses three synchronized input modalities:
- Text (from transcripts)
- Audio (vocal tones and prosody)
- Video (facial expressions and body language)

The trained ML model analyzes these inputs to produce:
- Emotion prediction (the primary emotional state)
- Sentiment prediction (the broader valence category)

The LLM (Large Language Model) is used separately. It does not make the prediction; instead, it provides a natural, conversational explanation of the ML model's prediction.

## SECTION 2 — EMOTION LABELS
Affectra AI predicts one of seven emotion classification labels. These are general observational categories, not absolute truths:
- **Anger**: Displays of frustration or hostility.
- **Disgust**: Displays of strong disapproval or revulsion.
- **Fear**: Displays of anxiety, alertness to danger, or panic.
- **Joy**: Displays of happiness, amusement, or intense pleasure.
- **Neutral**: A baseline state where no strong emotion is outwardly expressed.
- **Sadness**: Displays of grief, low energy, or disappointment.
- **Surprise**: Displays of sudden reaction to an unexpected event.

## SECTION 3 — SENTIMENT LABELS
Expressions are grouped into three broader sentiment categories:
- **Positive**: Optimistic, approving, or happy general valence.
- **Negative**: Pessimistic, painful, or disapproving general valence.
- **Neutral**: Expressions lacking strong positive or negative valence.

## SECTION 4 — EMOTION VS SENTIMENT
**Emotion classification** attempts to identify a specific, nuanced feeling (e.g., "Joy" vs "Surprise"). 
**Sentiment classification** is broader and identifies the overall polarity or "direction" of the expression (e.g., whether it is broadly positive, negative, or neutral).

## SECTION 5 — MODEL PREDICTIONS
Affectra predictions are probabilistic. The model outputs probabilities for all available classes. 
- A higher probability means stronger model confidence relative to other classes.
- A high probability does not guarantee correctness.
- Context can affect the true interpretation of an expression.
- Multimodal signals can sometimes be ambiguous (e.g., a person smiling out of nervousness).

## SECTION 6 — IMPORTANT LIMITATIONS
- Emotion classification can be ambiguous and subjective.
- Context matters; the model does not have access to the user's personal history or the broader situation.
- Predictions can be incorrect.
- Emotion classification is not a medical diagnosis.
- Emotion classification does not establish a mental-health condition.
- The system should not be used as a substitute for a qualified professional (e.g., a psychologist or psychiatrist).
- Different emotion classes may have different performance characteristics (some are easier for the model to detect than others).

## SECTION 7 — AFFECTRA MODEL ARCHITECTURE
The production model uses the following architecture:
Text features + Audio features + Video features
        ↓
Gated Multimodal Fusion
        ↓
Emotion + Sentiment

Each modality is represented using a 768-dimensional feature representation before fusion. The final Experiment 2 fusion dimension is 512.

## SECTION 8 — PREDICTION AND EXPLANATION SEPARATION
There is a strict separation of concerns in the system:
- **ML model**: Predicts the emotion and sentiment based on extracted multimodal features.
- **LLM**: Explains the prediction in human-readable language, without changing the prediction.
- **RAG (Retrieval-Augmented Generation)**: Will later provide relevant factual knowledge from this project to the LLM to help it explain predictions accurately. RAG does not perform emotion classification.
