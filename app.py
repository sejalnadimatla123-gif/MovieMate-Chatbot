

import streamlit as st
from chatbot import MovieChatbot

st.set_page_config(
    page_title="MovieMate 🎬",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 MovieMate")
st.caption("AI-powered movie discovery — TMDB 5000 + Google Gemini + FAISS")


if "bot" not in st.session_state:
    with st.spinner("Loading MovieMate (first load ~30s)..."):
        st.session_state.bot      = MovieChatbot()
        st.session_state.messages = []

bot = st.session_state.bot


with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **MovieMate** uses:
    - 🤖 **Google Gemini** — response generation
    - 🔍 **FAISS** — vector similarity search
    - 🧠 **Sentence Transformers** — embeddings
    - 🎬 **TMDB 5000** — movie dataset
    """)
    st.divider()
    st.markdown("**Try asking:**")
    examples = [
        "Suggest sci-fi movies after 2010",
        "Movies directed by Christopher Nolan",
        "Feel-good comedy movies",
        "Thriller movies with high ratings",
        "Movies similar to Inception",
        "Action movies under 2 hours",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state.pending = ex

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        bot.reset()
        st.rerun()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "Hi! I'm **MovieMate** 🎬 Ask me to recommend movies, "
            "find films by director or actor, or explore by genre. "
            "What are you in the mood for?"
        )


def handle_query(query: str):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            reply = bot.chat(query)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})



if "pending" in st.session_state:
    query = st.session_state.pop("pending")
    handle_query(query)
    st.rerun()


if prompt := st.chat_input("Ask about movies..."):
    handle_query(prompt)
