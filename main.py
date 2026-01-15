import streamlit as st
import asyncio
from pyrogram import Client
from transformers import pipeline

# --- Настройки ---
API_ID = 'ВАШ_API_ID'
API_HASH = 'ВАШ_API_HASH'

# --- Загрузка модели ---
@st.cache_resource
def load_model():
    # Модель rubert-tiny-toxicity очень быстрая и точная для русского языка
    return pipeline("text-classification", model="cointegrated/rubert-tiny-toxicity")

classifier = load_model()

# --- Функция сбора данных через Pyrogram ---
async def get_channel_comments(channel_link, post_limit=5):
    comments_text = []
    
    # Используем context manager для управления сессией
    async with Client("my_account", API_ID, API_HASH) as app:
        # Получаем информацию о канале/чате
        chat = await app.get_chat(channel_link)
        
        # Перебираем последние посты
        async for message in app.get_chat_history(chat.id, limit=post_limit):
            # В Telegram комментарии — это сообщения в связанном мегагрупп-чате
            try:
                # Пытаемся получить ветку обсуждения (replies)
                async for reply in app.get_discussion_replies(chat.id, message.id):
                    if reply.text:
                        comments_text.append(reply.text)
            except Exception:
                # Если обсуждение закрыто или недоступно
                continue
                
    return comments_text

# --- Интерфейс Streamlit ---
st.set_page_config(page_title="Toxicity Detector", layout="wide")

st.title("🛡️ Анализ токсичности аудитории")
st.markdown("Введите ссылку на публичный канал, чтобы оценить уровень агрессии в комментариях.")

url = st.text_input("Ссылка на канал", placeholder="https://t.me/example_channel")

if st.button("Начать анализ"):
    if url:
        with st.status("Сбор данных...", expanded=True) as status:
            try:
                # Просто используем asyncio.run для запуска асинхронной функции
                raw_comments = asyncio.run(get_channel_comments(url))
                
                if not raw_comments:
                    st.error("Комментарии не найдены.")
                else:
                    st.write(f"✅ Собрано комментариев: {len(raw_comments)}")
                    st.write("🧠 Анализируем текст...")
                    
                    # Классификация
                    predictions = classifier(raw_comments)
                    
                    # Обработка результатов
                    toxic_messages = [
                        (raw_comments[i], predictions[i]['label']) 
                        for i in range(len(predictions)) 
                        if predictions[i]['label'] != 'non-toxic'
                    ]
                    
                    status.update(label="Анализ завершен!", state="complete", expanded=False)

                    # Визуализация
                    toxic_count = len(toxic_messages)
                    total_count = len(raw_comments)
                    toxic_percent = (toxic_count / total_count) * 100

                    col1, col2 = st.columns(2)
                    col1.metric("Общий уровень токсичности", f"{toxic_percent:.1f}%")
                    col2.metric("Найдено негативных сообщений", toxic_count)

                    # Прогресс-бар
                    st.progress(toxic_percent / 100)

                    if toxic_messages:
                        with st.expander("Показать подозрительные комментарии"):
                            for text, label in toxic_messages:
                                st.warning(f"**[{label}]**: {text}")
                    else:
                        st.success("Токсичных комментариев не обнаружено!")

            except Exception as e:
                st.error(f"Ошибка доступа: {e}")
                st.info("Убедитесь, что ссылка верна и канал публичный.")
    else:
        st.warning("Введите ссылку!")