import streamlit as st
import pickle
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# Load model SVM dari pkl
with open("models/svm_imdb_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")
bert.eval()

# Function ubah teks jadi embedding
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = bert(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# Function prediksi sentimen
def predict_sentiment(text):
    vector = get_bert_embedding(text).reshape(1, -1)
    pred = model.predict(vector)[0]
    return "Positif 😊" if pred == 1 else "Negatif 😠"

# UI Streamlit
st.title("🎬 Analisis Sentimen Review Film")
st.write("Enter a movie review and see if it is positive or negative.")

user_input = st.text_area("Write here:")

if st.button("Analisis"):
    if user_input.strip() == "":
        st.warning("Please enter the review text first!")
    else:
        hasil = predict_sentiment(user_input)
        st.success(f"Sentiment Results: **{hasil}**")

if st.button("💬 Mau ngobrol lebih lanjut?"):
    st.switch_page("pages/ChatbotFilm.py")

