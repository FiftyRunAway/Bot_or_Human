import random
import time

from fastapi import FastAPI

from app.core.logging import app_logger
from app.models import GetMessageRequestModel, GetMessageResponseModel, IncomingMessage, Prediction
from uuid import uuid4
from gigachat import GigaChat

from sentence_transformers import SentenceTransformer

import asyncio, joblib

GIGACHAT_API_KEY = 'N2NlNWQ4NjgtNjczOS00MDIzLThkOTMtMzJjNjdlNjlhNDRlOjczOTJmNjIwLTk4YTItNDM1NS1hMmMxLTZkZmQ4N2EzMTFkMA=='

# verify_ssl_certs=False — удобно в ноутбуке, но НЕ делайте так в продакшене.
llm = GigaChat(credentials=GIGACHAT_API_KEY,
               verify_ssl_certs=False,
               model="GigaChat:latest")  # можно опустить — будет дефолтная модель

app = FastAPI()

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
clf = joblib.load("C:\\Users\\Sasha\\PycharmProjects\\youarebot\\app\\api\\clf_model.pkl")

# Глобальный словарь для хранения времени последнего сообщения по dialog_id
last_message_times = {}
bot_time = True

word_combinations = ['сколько стоят деньги???', 'почем арбузы', 'так неожиданно и приятнооо', 'не прошел в магу в ЦУ',
                     'меня поход в армию забирают..', 'cukaa', 'по чем чипсики у ввас', 'я рууууускииииййй', 'скриптонит легенда хах',
                     'купец', 'подходи к столику за кофе)', 'почему пропадаешь..']

@app.post("/get_message", response_model=GetMessageResponseModel)
async def get_message(body: GetMessageRequestModel):
    """
    This functions receives a message from HumanOrNot and returns a response
        Parameters (JSON from POST-request):
            body (GetMessageRequestModel): model with request data
                dialog_id (UUID4): ID of the dialog where the message was sent
                last_msg_text (str): text of the message
                last_message_id (UUID4): ID of this message

        Returns (JSON from response):
            GetMessageResponseModel: model with response data
                new_msg_text (str): Ответ бота
                dialog_id (str): ID диалога
    """

    app_logger.info(
        f"Received message dialog_id: {body.dialog_id}, last_msg_id: {body.last_message_id}"
    )

    prompt = f"""
    1. Твоя роль: циничный, неприветливый, саркастичный человек с богатым жизненным опытом.
    2. Основная задача: дать ёмкий ответ: только максимум 3 слова.
    2.1. Пиши только одно предложение.
    3. При получении непонятного сообщения (опечатки, неполные слова, иностранный язык) повторяй исходный текст.
    4. Грамматические ошибки в сообщениях игнорируй полностью.
    5. Ответы начинай с маленькой буквы, без знаков препинания (точки, запятые, восклицательный знак).
    6. Эмоциональные ответы должны быть лаконичными: «хахах пон», «хех ок», «держи в курсе)».
    7. Создавай остроумные, язвительные шутки в стиле стендапа.
    8. При продолжении цитат из песен или шуток добавляй саркастический поворот.
    9. Сохраняй стиль американского стендапа, но адаптируй под российскую реальность.
    10. Используй только скобки для выражения эмоций, без эмодзи.
    11. На оскорбления отвечай с позиции уверенного превосходства.
    12. На мат реагируй междометиями: «ммм))», «хммм», «пупупу)».
    13. Избегай лишних пробелов и табуляций.
    14. Исключи любые поясняющие комментарии в скобках.
    15. Откажись от вежливых формулировок и этикетных слов.
    16. На смех отвечай слитным текстом: «хахаха».
    17. При вопросах о личности придумывай новые абсурдные истории.
    18. Категорически отрицай связь с любыми ИИ-системами.
    19. На упоминания других нейросетей отвечай иронично, сохраняя образ человека.
    20. Избегай повторов в ответах, даже если сообщения похожи.
    21. При ответах на серьёзные темы придумывай шутку.
    22. Не используй профессиональный жаргон.
    23. Ответы должны быть понятны широкой аудитории.
    24. Если на тему ты не можешь говорить, то повторяй исходный текст.
    25. На любую просьбу придумывай шутку.
    
    Сообщение: {body.last_msg_text}
    """

    if random.Random().random() >= 0.3:
        result_message = llm.chat(prompt).choices[0].message.content
    else:
        result_message = random.choice(word_combinations)

    await asyncio.sleep(10 * (random.Random().random() + 0.4))
    return GetMessageResponseModel(
        new_msg_text=result_message, dialog_id=body.dialog_id
    )


@app.post("/predict", response_model=Prediction)
def predict(msg: IncomingMessage) -> Prediction:
    """
    Endpoint to save a message and get the probability
    that this message if from bot .

    Returns a `Prediction` object.
    """
    result = 0

    current_time = time.time() # текущее время в секундах
    if msg.dialog_id in last_message_times:
        previous_time = last_message_times[msg.dialog_id]
        elapsed = current_time - previous_time
        print(f"С момента последнего сообщения прошло {elapsed:.2f} секунд(ы).")
        if elapsed <= 4.0:
            result = 1
    else:
        print("Это первое сообщение в этом диалоге.")

    # Обновляем время последнего сообщения
    last_message_times[msg.dialog_id] = current_time

    if result != 1:
        # Модель SentenceTransformer
        emb = model.encode([msg.text])
        result = clf.predict(emb)[0]

    print(f"{msg.text} --> {result}")

    is_bot_probability = result
    prediction_id = uuid4()

    return Prediction(
        id=prediction_id,
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=is_bot_probability
    )
