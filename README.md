# NLP_project_Emotional_mesages
🎭 Emotion Detection using BiLSTM
📌 Overview

This project performs Emotion Detection on text data using a Bidirectional LSTM (BiLSTM) model.
The dataset contains 416,809 text samples labeled with six emotions:

😢 Sadness

😀 Joy

❤️ Love

😡 Anger

😨 Fear

😲 Surprise

The model achieves ~94% accuracy and is deployed as a Streamlit app for real-time predictions.

📂 Project Structure
emotion_app/
│── app.py                # Streamlit app
│── best_bilstm.h5        # Trained BiLSTM model
│── tokenizer.pkl         # Tokenizer (for preprocessing text)
│── requirements.txt      # Project dependencies
│── README.md             # Project documentation

⚙️ Installation

Clone this repository:

git clone https://github.com/yourusername/emotion-detection-bilstm.git
cd emotion-detection-bilstm


Create a virtual environment and install dependencies:

pip install -r requirements.txt


Make sure you have the model and tokenizer:

best_bilstm.h5 → trained model

tokenizer.pkl → tokenizer object

📊 Model Details

Preprocessing: spaCy lemmatization, stopword removal, Keras tokenization, padding (maxlen=20)

Model:

Embedding layer (vocab=10k, dim=100)

BiLSTM (128 units, dropout=0.2)

BatchNormalization + Dropout

Dense(64, relu)

Dense(6, softmax)

Metrics: Accuracy = ~94%, Macro F1 = 0.92

✨ Example Usage

Input:

"I feel so anxious about my exams tomorrow"


Output:

Predicted Emotion: Fear

🔮 Future Work

Add pre-trained embeddings (GloVe / Word2Vec).

Experiment with attention mechanism.

Improve handling of class imbalance.

Extend app with visual analytics (word clouds, charts).

👨‍💻 Author: Sahil Kasana
