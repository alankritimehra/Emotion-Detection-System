import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.title("Emotion Detection System")
st.write("Enter text and the model will detect emotional and psychological patterns.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("./emotion_model/emotion_model")
    model = AutoModelForSequenceClassification.from_pretrained("./emotion_model/emotion_model")
    model.to(device)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

labels = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

state_map = {
    "joy": "Positive / Happy",
    "excitement": "Positive / Excited",
    "gratitude": "Positive / Thankful",
    "optimism": "Hopeful / Positive",
    "relief": "Relieved",
    "love": "Affectionate",
    "fear": "Fearful / Anxious",
    "nervousness": "Anxious",
    "anger": "Angry / Irritated",
    "annoyance": "Slightly Frustrated",
    "sadness": "Sad",
    "grief": "Grieving",
    "curiosity": "Curious",
    "confusion": "Confused",
    "surprise": "Surprised",
    "approval": "Positive",
    "admiration": "Respect / Admiration",
    "amusement": "Amused",
    "caring": "Compassionate",
    "desire": "Motivated / Desire",
    "disappointment": "Disappointed",
    "disapproval": "Negative Opinion",
    "disgust": "Disgusted",
    "embarrassment": "Embarrassed",
    "pride": "Proud",
    "realization": "Reflective",
    "remorse": "Regretful",
    "neutral": "Neutral"
}
text = st.text_area("Enter your text:")

threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.30)

if st.button("Detect Emotion"):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits)[0]

        results = []

        for i, prob in enumerate(probs):
            if prob.item() >= threshold:
                emotion = labels[i]
                state = state_map.get(emotion, "Emotional Pattern Detected")
                results.append((emotion, state, prob.item()))

        results = sorted(results, key=lambda x: x[2], reverse=True)

        if results:
            st.subheader("Detected Emotions")
            for emotion, state, confidence in results:
                st.write(f"**Emotion:** {emotion}")
                st.write(f"**State:** {state}")
                st.write(f"**Confidence:** {confidence:.4f}")
                st.write("---")
        else:
            st.info("No strong emotion detected. Try lowering the threshold.")

st.caption("Note: This model detects emotional patterns in text. It is not a medical or psychological diagnosis.")