import pandas, json, os, sklearn
import joblib
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer

# === Загрузка данных ===
with open("train_with_is_bot.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = []
labels = []

# Идём по всем диалогам и собираем отдельные сообщения
for dialog in data.values():
    for msg in dialog:
        texts.append(msg["text"])
        labels.append(int(msg["is_bot"]))  # 0 - человек, 1 - бот

print(f"Всего сообщений: {len(texts)}")

# === Быстрые sentence embeddings ===
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  # Поддерживает русский и английский

embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

# === Обучение классификатора ===
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred, target_names=["человек", "бот"]))

joblib.dump(clf, "../api/clf_model.pkl")

def classify_text(text):
    emb = model.encode([text])
    pred = clf.predict(emb)[0]
    return "бот" if pred == 1 else "человек"

# 🔍 Пример использования
sample_text = "Привет! Чем могу помочь тебе сегодня?"
result = classify_text(sample_text)
print(f"Текст: {sample_text} → {result}")