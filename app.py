import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer and model
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model("best_bilstm.h5")

# Parameters
MAX_LEN = 20   # same as training
labels_map = {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}

# Streamlit UI
st.title("🎭 Emotion Detection with BiLSTM")
st.write("Enter text below and the model will predict its emotion:")

user_input = st.text_area("Enter a sentence:", "")

if st.button("Predict Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Convert text to sequence
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

        # Predict
        probs = model.predict(padded)
        pred_class = np.argmax(probs)

        # Show results
        st.success(f"Predicted Emotion: **{labels_map[pred_class]}**")
        st.write("Confidence scores:")
        for i, label in labels_map.items():
            st.write(f"{label}: {probs[0][i]:.4f}")
