

import os
import warnings
import pandas as pd
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from vectorstore import load_vectorstore, retrieve

warnings.filterwarnings("ignore")
load_dotenv()

MODEL_NAME   = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
MAX_HISTORY  = 6

SYSTEM_PROMPT = """You are MovieMate, a friendly and knowledgeable movie recommendation assistant.
Your job is to help users discover movies based on their preferences.

When given a list of retrieved movies, craft a natural, engaging response that:
- Mentions the movie title, year, genre, and rating
- Gives a brief 1-sentence reason why it suits the user's request
- Groups or ranks them naturally if relevant

Rules:
- Only recommend movies from the retrieved list — do not invent titles
- If the user follows up or refines their query, use conversation history for context
- Keep responses concise and conversational
- If no good matches exist, say so and suggest the user rephrase their query"""


class MovieChatbot:
    def __init__(self):
        print("Loading vector store...")
        self.index, self.df, self.texts = load_vectorstore()

        print(f"Loading embedding model: {MODEL_NAME}")
        self.embed_model = SentenceTransformer(MODEL_NAME)

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found.\n"
                "Add it to your .env file: GOOGLE_API_KEY=your_key_here\n"
                "Get a free key at: https://aistudio.google.com"
            )

        print(f"Connecting to Gemini ({GEMINI_MODEL})...")
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0.7,
            convert_system_message_to_human=True,
        )

        self.history = []
        print("MovieMate ready!\n")

    def _format_retrieved(self, movies: pd.DataFrame) -> str:
        
        lines = []
        for _, row in movies.iterrows():
            year    = int(row["year"])    if pd.notna(row.get("year"))    else "N/A"
            rating  = f"{float(row['rating']):.1f}" if pd.notna(row.get("rating"))  else "N/A"
            runtime = f"{int(row['runtime'])} min"  if pd.notna(row.get("runtime")) and str(row.get("runtime")) != "" else "N/A"
            lines.append(
                f"- {row['title']} ({year}) | "
                f"Genre: {row['genre']} | "
                f"Rating: {rating}/10 | "
                f"Director: {row['director']} | "
                f"Cast: {row['cast']} | "
                f"Runtime: {runtime} | "
                f"Summary: {str(row['summary'])[:200]}"
            )
        return "\n".join(lines)

    def _build_messages(self, augmented_query: str) -> list:
        
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for turn in self.history[-MAX_HISTORY:]:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                messages.append(AIMessage(content=turn["content"]))
        messages.append(HumanMessage(content=augmented_query))
        return messages

    def chat(self, user_query: str) -> str:
        
        retrieved = retrieve(
            user_query, self.embed_model, self.index, self.df, top_k=5
        )
        context = self._format_retrieved(retrieved)

        augmented = (
            f"User question: {user_query}\n\n"
            f"Retrieved movies from the database:\n{context}\n\n"
            "Based on the movies above and the conversation history, "
            "provide a helpful, natural response."
        )

        messages = self._build_messages(augmented)
        response = self.llm.invoke(messages)
        reply    = response.content.strip()

        self.history.append({"role": "user",      "content": user_query})
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self):
        
        self.history = []
        print("Conversation cleared.")


def run_terminal():
    
    bot = MovieChatbot()
    print("Type 'quit' to exit, 'reset' to clear history.\n")
    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() == "quit":
            break
        if query.lower() == "reset":
            bot.reset()
            continue
        reply = bot.chat(query)
        print(f"\nMovieMate: {reply}\n")


if __name__ == "__main__":
    run_terminal()
