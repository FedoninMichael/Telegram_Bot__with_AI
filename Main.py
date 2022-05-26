import re
import nltk
import random
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from telegram import Update  # Кусок входящей информации от сервера Телеграм
from telegram.ext import Filters, Updater # Инструмент, который позволяет получать Апдейты
from telegram.ext import MessageHandler # Обработчик = функция, которая будет вызы

config_file = open("big_bot_config.json", "r")
BIG_CONFIG = json.load(config_file)

def filter_text(text):
    text = text.lower() # Приводим к нижнему регистру ("ПрИвет" => "привет")
    pattern = r'[^\w\s]' # Удаление знаков препинания, Удалить все кроме букв и пробелов
    text = re.sub(pattern, "", text) # Делать замену символов в строке
    return text

# На вход: два текста, на выход: boolean(True, False)
# Функция isMatch вернет True, если тексты совпадают или False иначе
def isMatch(text1, text2):
    text1 = filter_text(text1)
    text2 = filter_text(text2)

    # Проверить что одна фраза является частью другой
    if len(text1) == 0 or len(text2) == 0:
        return False

    # Text1 содержит text2
    if text1.find(text2) != -1:
        return True

    # Text2 содержит text1
    if text2.find(text1) != -1:
        return True

    # Расстояние Левенштейна (edit distance = расстояние редактирования)
    distance = nltk.edit_distance(text1, text2)  # Кол-во символов 0...Inf
    length = (len(text1) + len(text2)) / 2  # Средняя длина двух текстов
    score = distance / length  # 0...1
    return score < 0.3

    return text1 == text2

# Намерения = Intents
# Поздароваться, спросить как дела, спросить имя или чем занимаешься
# Заказать пиццу, отменить заказ, добавить больше сыра
# Конфигурация бота
BOT_CONFIG = {
    # Все намерения которые поддерживает наш бот
    "intents": {
        "hello": {
            "examples" : ["Привет", "Здарова", "Йо", "Приветос", "Хеллоу"],
            "responses": ["Здравстсвтсвтвтвуй человек", "И тебе не хворать", "Здоровее видали"],
        },
        "how_are_you": {
            "examples" : ["Как дела", "Чо каво", "Как поживаешь"],
            "responses": ["Маюсь Фигней", "Веду интенсивы", "Учу Пайтон"],
        }
    },
    # Фразы когда бот не может ответить
    "failure_phrases": ["Даже не знаю что сказать", "Поставлен в тупик", "Перефразируйте, я всего лишь бот"],
}

examples = ["Привет", "Здарова", "Йо", "Приветос", "Хеллоу"]
responses = ["Здравстсвтсвтвтвуй человек", "И тебе не хворать", "Здоровее видали"]

text = input()
def printAnswer(text, examples, responses):
    for example in examples:  # Для каждого элемента списка examples
        if isMatch(text, example):  # Если пример совпадает с текстом пользователя
            print(random.choice(responses))  # Выводим на экран случайный элемент списка responses
            break

# Для каждого намерения, пытаемся напечатать ответ
for intent in BOT_CONFIG["intents"].values():
    printAnswer(text, intent["examples"], intent["responses"])

# Сколько символов нужно отредактировать "мама" = (1) => "мамы"
nltk.edit_distance("Привет братан", "Превед бротан")
# 0, 1, 2, 3...
# 0 ... 0.3 ... 0.5 ... 1


X = [] # Фразы
y = [] # Намерения
# Собираем фразы и интенты из BIG_CONFIG в X,y
for name, intent in BIG_CONFIG["intents"].items():
    for example in intent["examples"]:
        X.append(example)
        y.append(name)

# Подготовка данных к обучению модели
# NLP = Natural Language Processing

# Векторайзер = превращает тексты в наборы чисел (вектора)

# Мама круто мыла раму => [1,2,3,4]
# Круто мама раму мыла => [2,1,4,3]
# Мыла мама круто раму => [3,1,2,4]

vectorizer = CountVectorizer()  # Можно указать настройки
vectorizer.fit(X)  # Учится вот эти конкретные тексты превращать в числа

vectorizer.transform(["как твои дела друг"]).toarray()

# Классификация текста = предсказания класса(интент) по тексту(фразе)
#model = LogisticRegression() # Настройки
#vecX = vectorizer.transform(X) # Преобразуем тексты в вектора
#model.fit(vecX, y)  # Обучаем модель
#print(model.score(vecX, y))
# Насколько качественно обучилась модель?
# Пробуем разные модели, разные настройки

#model = RandomForestClassifier() # Настройки
#vecX = vectorizer.transform(X) # Преобразуем тексты в вектора
#model.fit(vecX, y)  # Обучаем модель
#print(model.score(vecX, y))

model = MLPClassifier(max_iter=500, hidden_layer_sizes=(100,100,100,)) # Настройки
vecX = vectorizer.transform(X) # Преобразуем тексты в вектора
model.fit(vecX, y)  # Обучаем модель
print(model.score(vecX, y))

def get_intent_ml(text):
    vec_text = vectorizer.transform([text])
    intent = model.predict(vec_text)[0]
    # ToDo: выдавать ответ только если модель уверена в нем
    return intent


def get_intent(text):
    for name, intent in BIG_CONFIG["intents"].items():
        for example in intent["examples"]:
            if isMatch(text, example):
                return name
    return None

def bot(phrase):
    phrase = filter_text(phrase)
    # Напрямую найти ответ
    intent = get_intent(phrase)
    # Если напрямую интент не нашелся
    if not intent:
        # Попросить модель найти ответ
        intent = get_intent_ml(phrase)
    # Если намерение определено
    if intent:
        responses = BIG_CONFIG["intents"][intent]["responses"]  # Находим варианты ответов
        return random.choice(responses)
    failure = BIG_CONFIG["failure_phrases"]
    return random.choice(failure)
    # Выдать Failure Phrase

msg = ""
exit_phrases = ["Выйти", "Выключись", "Bye", "Пока"]

while not msg in exit_phrases:
    msg = input()
    print("[BOT]: " + bot(msg))

def bot_telegram_reply(update: Update, ctx):
    text = update.message.text
    reply = bot(text)
    update.message.reply_text(reply)
    print(f"{text}: {reply}")

BOT_KEY = '5323508311:AAEsignJVivMUYnky1aFd7AVZZdxKiTHefM'
upd = Updater(BOT_KEY) # Создаем подключение к серверу ТГ, и будем ждать от него обновлений

# Создать MessageHandler
handler = MessageHandler(Filters.text, bot_telegram_reply)  # Текстовые сообщения будем обрабатывать этой функцией

# Зарегистрировать его в Апдейтере
upd.dispatcher.add_handler(handler)

# Начать опрос (polling) сервера ТГ
upd.start_polling()
upd.idle()







