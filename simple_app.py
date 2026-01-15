import streamlit as st
import nest_asyncio
import asyncio
from telethon.sync import TelegramClient

from toxic_model import load_toxicity_model

# ЭТОТ БЛОК РЕШАЕТ ВАШУ ОШИБКУ
nest_asyncio.apply()

# --- Настройки ---
API_ID = "37840327"
API_HASH = "277f6d284a5a61d73740be67e1dcee00"


@st.cache_resource
def get_model():
    # Загружаем нашу дообученную модель токсичности
    return load_toxicity_model()


tox_model = get_model()

st.title("🛡️ Проверка токсичности аудитории TG-канала")

channel_url = st.text_input("Ссылка на канал", "https://t.me/durov")

if st.button("Проверить"):
    if not channel_url:
        st.error("Введите ссылку!")
    else:
        with st.spinner("Работаем..."):
            try:
                # Явно создаем цикл событий для этого потока Streamlit
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                with TelegramClient("session_simple", API_ID, API_HASH) as client:
                    entity = client.get_entity(channel_url)
                    messages = client.get_messages(entity, limit=5)

                    comments_list = []
                    for msg in messages:
                        if msg.replies:
                            # Ограничим до 10 комментариев для скорости теста
                            for reply in client.iter_messages(
                                entity, reply_to=msg.id, limit=100
                            ):
                                if reply.message:
                                    comments_list.append(reply.message)

                    if not comments_list:
                        st.warning("Комментарии не найдены.")
                    else:
                        st.info(
                            f"Собрано {len(comments_list)} комментариев. Анализируем..."
                        )

                        # Используем нашу дообученную модель
                        results = tox_model.predict(comments_list)

                        bad_messages = []
                        for i, res in enumerate(results):
                            if res["label"] != "non-toxic":
                                bad_messages.append(
                                    {
                                        "Текст": comments_list[i][:100] + "...",
                                        "Тип": res["label"],
                                        "Оценка": f"{res['score']:.2f}",
                                    }
                                )

                        tox_level = (len(bad_messages) / len(comments_list)) * 100
                        st.metric("Уровень токсичности", f"{tox_level:.1f}%")

                        if bad_messages:
                            st.write("### Примеры подозрительных сообщений:")
                            st.table(bad_messages)
                        else:
                            st.success("Все чисто!")

            except Exception as e:
                st.error(f"Произошла ошибка: {e}")