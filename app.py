import streamlit as st
import numpy as np
import re
import pickle
import pandas as pd
import nltk
import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Analisis Sentimen Roblox',
    page_icon='🎮',
    layout='centered'
)

MAX_LEN = 100

# ── Load model & objek pendukung ─────────────────────────────────────────────
@st.cache_resource
def load_resources():
    model      = load_model('model_cnn_lstm.h5')
    with open('tokenizer_dl.pkl', 'rb') as f:
        tokenizer  = pickle.load(f)
    kamus      = pd.read_excel('kamuskatabaku.xlsx')
    slang_dict = dict(zip(kamus['tidak_baku'], kamus['kata_baku']))
    factory    = StemmerFactory()
    stemmer    = factory.create_stemmer()
    stop_words = set(stopwords.words('indonesian'))
    regexp     = RegexpTokenizer(r'\w+')
    return model, tokenizer, slang_dict, stemmer, stop_words, regexp

model, tokenizer, slang_dict, stemmer, stop_words, regexp = load_resources()

# ── Preprocessing ────────────────────────────────────────────────────────────
def preprocess(text):
    text = re.sub(r'https\S+', ' ', str(text), flags=re.IGNORECASE)
    text = text.lower()
    text = re.sub(r'[@#]\S+', ' ', text)
    text = re.sub(r"'\w+", ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join([slang_dict.get(w, w) for w in text.split()])
    text = ' '.join([w for w in text.split() if w not in stop_words])
    tokens = regexp.tokenize(text)
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join([w for w in tokens if len(w) > 2])

# ── Prediksi ─────────────────────────────────────────────────────────────────
label_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}
emoji_map = {'Negatif': '😡', 'Netral': '😐', 'Positif': '😊'}
color_map = {'Negatif': '#ff4b4b', 'Netral': '#f0a500', 'Positif': '#21c354'}

def predict(text):
    bersih = preprocess(text)
    seq    = tokenizer.texts_to_sequences([bersih])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    proba  = model.predict(padded, verbose=0)[0]
    idx    = int(np.argmax(proba))
    label  = label_map[idx]
    return label, proba

# ── UI ────────────────────────────────────────────────────────────────────────
st.title('🎮 Analisis Sentimen Ulasan Roblox')
st.caption('Model: CNN+LSTM · Dataset: 10.000+ ulasan Google Play Store')
st.divider()

tab1, tab2 = st.tabs(['Analisis Teks', 'Analisis Batch (CSV)'])

# Tab 1 — Input tunggal
with tab1:
    teks = st.text_area('Masukkan ulasan:', height=120,
                        placeholder='Contoh: game ini seru banget, tapi sering lag...')
    if st.button('Prediksi', use_container_width=True):
        if teks.strip():
            with st.spinner('Menganalisis...'):
                label, proba = predict(teks)
            st.markdown(f'### Hasil: {emoji_map[label]} **:{label.lower()}[{label}]**')
            st.markdown('**Probabilitas per kelas:**')
            col1, col2, col3 = st.columns(3)
            col1.metric('😡 Negatif', f'{proba[0]*100:.1f}%')
            col2.metric('😐 Netral',  f'{proba[1]*100:.1f}%')
            col3.metric('😊 Positif', f'{proba[2]*100:.1f}%')
        else:
            st.warning('Masukkan teks ulasan terlebih dahulu.')

# Tab 2 — Batch CSV
with tab2:
    st.markdown('Upload file CSV dengan kolom **`ulasan`** berisi teks yang ingin dianalisis.')
    uploaded = st.file_uploader('Upload CSV', type=['csv'])
    if uploaded:
        df_up = pd.read_csv(uploaded)
        if 'ulasan' not in df_up.columns:
            st.error("Kolom 'ulasan' tidak ditemukan di CSV.")
        else:
            with st.spinner('Menganalisis semua ulasan...'):
                results = df_up['ulasan'].apply(lambda x: predict(x)[0])
                df_up['sentimen'] = results
            st.success(f'Selesai! {len(df_up)} ulasan dianalisis.')
            st.dataframe(df_up[['ulasan', 'sentimen']], use_container_width=True)

            csv_out = df_up.to_csv(index=False).encode('utf-8')
            st.download_button('Download Hasil CSV', csv_out,
                               file_name='hasil_sentimen.csv', mime='text/csv')
