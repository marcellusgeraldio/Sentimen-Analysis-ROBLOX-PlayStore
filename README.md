# 🎮 Sentiment Analysis – Roblox Google Play Store Reviews

Proyek analisis sentimen ulasan aplikasi Roblox di Google Play Store menggunakan deep learning.

## 📋 Deskripsi

Model ini mengklasifikasikan ulasan pengguna ke dalam tiga kelas sentimen:
- 😊 **Positif** — ulasan dengan rating 4–5
- 😐 **Netral** — ulasan dengan rating 3
- 😡 **Negatif** — ulasan dengan rating 1–2

## 🗂️ Struktur Proyek

```
sentiment-roblox/
├── app.py                        # Streamlit web app
├── Sentimen_Analisis_Roblox.ipynb  # Notebook utama (scraping, preprocessing, training)
├── Inference_Sentimen_Roblox.ipynb # Notebook inference
├── requirements.txt              # Dependencies
├── model_cnn_lstm.h5             # Model terbaik (CNN+LSTM)
├── tokenizer_dl.pkl              # Tokenizer Keras
└── kamuskatabaku.xlsx            # Kamus normalisasi slang
```

## 🧠 Model & Skema Pelatihan

| Model | Ekstraksi Fitur | Split | Train Acc | Test Acc |
|-------|----------------|-------|-----------|----------|
| CNN+LSTM | Embedding | 80/20 | >92% | >92% |
| Bidirectional LSTM | Embedding | 80/20 | >92% | >92% |
| SVM | TF-IDF | 80/20 | — | >85% |
| Random Forest | Word2Vec | 80/20 | — | >85% |
| Random Forest | TF-IDF | 70/30 | — | >85% |

## ⚙️ Tech Stack

- **Scraping**: google-play-scraper
- **NLP**: NLTK, Sastrawi, Gensim (Word2Vec)
- **ML/DL**: TensorFlow/Keras, Scikit-learn
- **Deployment**: Streamlit

## 🚀 Cara Menjalankan Lokal

```bash
git clone https://github.com/username/sentiment-roblox.git
cd sentiment-roblox
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Demo

([Link Streamlit App](https://huggingface.co/spaces/Kichift/Sentimen-Analysis-ROBLOX))
