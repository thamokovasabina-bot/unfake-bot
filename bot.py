import os
import joblib
import numpy as np
import re
import pymorphy3
import spacy
import nltk
import telebot
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

nltk.download('stopwords', quiet=True)

MODEL_PATH = "best_model.joblib"
W2V_PATH = "word2vec_ruscorpora.kv"

classifier = joblib.load(MODEL_PATH)
print("Модель успешно загружена")

morph = pymorphy3.MorphAnalyzer()
nlp = spacy.load("ru_core_news_sm")
wv = KeyedVectors.load(W2V_PATH)

stop_words = set(stopwords.words('russian')) - {
    'более','больше','всегда','иногда','лучше','нет','не','хорошо','никогда'
}

class TextPreprocessing(BaseEstimator, TransformerMixin):
    def fit(self, corpus):
        return self

    def transform(self, corpus):
        return [self.clean_text(text) for text in corpus]

    def clean_text(self, text):
        text = text.lower()
        tokens = re.findall(r'[а-яё]+', text)
        tokens = [morph.parse(t)[0].normal_form for t in tokens]
        tokens = [t for t in tokens if t not in stop_words]
        return ' '.join(tokens)

class W2VVectorizer(BaseEstimator, TransformerMixin):
    def fit(self, corpus):
        return self

    def transform(self, corpus):
        return np.array([self.vec_text(text) for text in corpus])

    def vec_text(self, text):
        vectors_word = []
        doc = nlp(text)
        text_markup = [f'{token.text}_{token.pos_}' for token in doc]

        for word in text_markup:
            if word in wv.index_to_key:
                vectors_word.append(wv[word])
            else:
                vectors_word.append(np.zeros(300))

        return np.mean(vectors_word, axis=0)

model_fake_analysis = Pipeline([
    ('preproc', TextPreprocessing()),
    ('vec', W2VVectorizer()),
    ('clf', classifier)
])

bot = telebot.TeleBot(os.getenv("BOT_TOKEN"))

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, "Бот запущен! Отправь мне текст новости")

@bot.message_handler(content_types=["text"])
def handle_text(message):
    pred = model_fake_analysis.predict([message.text])[0]
    proba = model_fake_analysis.predict_proba([message.text])[0]
    confidence = max(proba) * 100

    label = "фейк" if pred == 1 else "не фейк"

    bot.send_message(
        message.chat.id,
        f"Новость:\n{message.text}\n\nРезультат: {label}\nУверенность: {confidence:.1f}%",
    )

bot.polling(non_stop=True)
