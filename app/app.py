import streamlit as st
import nest_asyncio
import asyncio
from telethon.sync import TelegramClient

from src.toxic_model import load_toxicity_model

nest_asyncio.apply()

API_ID = "34929851"
API_HASH = "8e89fcadcf6eeff26c6aa18cc686d96a"

st.set_page_config(
    page_title="Анализ токсичности Telegram",
    page_icon="🧪",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    * {
        box-sizing: content-box !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_model():
    return load_toxicity_model()

tox_model = get_model()

st.markdown("# :green[☣︎] Анализ токсичности аудитории", )
st.markdown("Оцените уровень токсичности аудитории в Telegram-канале")

# Основная форма
with st.container():
    st.markdown("### Параметры анализа")
    
    channel_username = st.text_input(
        "**Username канала**",
        placeholder="durov",
        help="Введите username канала без @ (например: durov)"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Количество постов**")
        post_limit_option = st.radio(
            "Выберите количество постов для анализа:",
            ['10', '50', '100', '200', '500', 'Все'],
            horizontal=False,
            help="Чем больше постов, тем точнее анализ, но дольше обработка"
        )
    
    with col2:
        st.markdown("**Комментариев на пост**")
        comment_limit_option = st.radio(
            "Выберите количество комментариев:",
            ['10', '50', '100', '200', '500', '1000', 'Все'],
            horizontal=False,
            help="Количество комментариев, которые будут проанализированы из каждого поста"
        )
    
    post_limit = None if post_limit_option == 'Все' else int(post_limit_option)
    comment_limit = None if comment_limit_option == 'Все' else int(comment_limit_option)
    
    if post_limit and comment_limit:
        estimated_comments = post_limit * comment_limit
        st.info(f"Будет проанализировано примерно **{estimated_comments}** комментариев")
    elif post_limit:
        st.info(f"Будет проанализировано **{post_limit}** постов (количество комментариев зависит от активности в канале)")
    elif comment_limit:
        st.info(f"Из каждого поста будет проанализировано **{comment_limit}** комментариев")
    else:
        st.info("Будет проанализированы все доступные посты и комментарии (это может занять много времени)")

st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    analyze_button = st.button("Начать анализ", use_container_width=True, type="primary")

if analyze_button:
    if not channel_username:
        st.error("Пожалуйста, введите username канала!")
    else:
        with st.spinner("Подключаемся к Telegram и собираем данные..."):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                with TelegramClient("session_simple", API_ID, API_HASH) as client:
                    entity = client.get_entity(f"https://t.me/{channel_username}")
                    messages = client.get_messages(entity, limit=post_limit)

                    comments_list = []
                    for msg in messages:
                        if msg.replies:
                            for reply in client.iter_messages(
                                entity, reply_to=msg.id, limit=comment_limit
                            ):
                                if reply.message:
                                    comments_list.append(reply.message)

                    if not comments_list:
                        st.warning("Комментарии не найдены. Убедитесь, что в канале есть посты с комментариями.")
                    else:
                        st.success(f"Собрано **{len(comments_list)}** комментариев из **{len(messages)}** постов")
                        
                        st.markdown("### Анализ комментариев")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        try:
                            batch_size = tox_model.batch_size
                            total_batches = (len(comments_list) + batch_size - 1) // batch_size
                            
                            def update_progress(current: int, total: int):
                                progress = current / total
                                progress_bar.progress(progress)
                            
                            status_text.text("Начинаем обработку комментариев...")
                            results = tox_model.predict(comments_list, progress_callback=update_progress)
                            progress_bar.progress(1.0)
                            status_text.text("Обработка завершена!")
                        except Exception as e:
                            st.error(f"Ошибка при обработке модели: {e}")
                            st.info("Попробуйте уменьшить количество постов или комментариев.")
                            import traceback
                            with st.expander("Детали ошибки"):
                                st.code(traceback.format_exc())
                            raise

                        # Анализ результатов
                        bad_messages = []
                        for i, res in enumerate(results):
                            if res["label"] != "non-toxic":
                                categories_str = ", ".join(res["categories"])
                                bad_messages.append(
                                    {
                                        "Текст": comments_list[i],
                                        "Категории": categories_str,
                                        "Вероятность": f"{res['max_probability']:.1%}",
                                    }
                                )

                        tox_level = (len(bad_messages) / len(comments_list)) * 100
                        
                        # Отображение результатов
                        st.markdown("---")
                        st.markdown("### Результаты анализа")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Всего комментариев",
                                f"{len(comments_list)}",
                                help="Общее количество проанализированных комментариев"
                            )
                        
                        with col2:
                            st.metric(
                                "Токсичных найдено",
                                f"{len(bad_messages)}",
                                delta=f"{len(bad_messages) - len(comments_list) // 10}",
                                delta_color="inverse",
                                help="Количество комментариев, классифицированных как токсичные"
                            )
                        
                        with col3:
                            if tox_level >= 20:
                                color_class = "toxicity-high"
                                emoji = "🔴"
                            elif tox_level >= 10:
                                color_class = "toxicity-medium"
                                emoji = "🟡"
                            else:
                                color_class = "toxicity-low"
                                emoji = "🟢"
                            
                            st.metric(
                                f"{emoji} Уровень токсичности",
                                f"{tox_level:.1f}%",
                                help="Процент токсичных комментариев от общего количества"
                            )

                        if bad_messages:
                            st.markdown("---")
                            st.markdown("### Обнаруженные токсичные комментарии")
                            st.dataframe(
                                bad_messages,
                                width="stretch",
                                hide_index=True,
                                column_config={
                                    "Текст": st.column_config.TextColumn(
                                        "Текст комментария",
                                        width="large"
                                    ),
                                    "Категории": st.column_config.TextColumn(
                                        "Категории токсичности",
                                        width="medium"
                                    ),
                                    "Вероятность": st.column_config.TextColumn(
                                        "Вероятность",
                                        width="small"
                                    ),
                                }
                            )
                            
                            category_counts = {}
                            for msg in bad_messages:
                                categories = msg["Категории"].split(", ")
                                for cat in categories:
                                    category_counts[cat] = category_counts.get(cat, 0) + 1
                            
                            if category_counts:
                                st.markdown("#### Распределение по категориям")
                                cat_cols = st.columns(len(category_counts))
                                for idx, (cat, count) in enumerate(category_counts.items()):
                                    with cat_cols[idx]:
                                        st.metric(cat, count)
                        else:
                            st.markdown("---")
                            st.success("Отлично! Токсичные комментарии не обнаружены. Аудитория канала выглядит здоровой.")

            except Exception as e:
                st.error(f"Произошла ошибка: {e}")
                st.info("Проверьте правильность username канала и убедитесь, что он открыт.")