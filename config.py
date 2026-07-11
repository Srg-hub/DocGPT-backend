import os
from dotenv import load_dotenv

load_dotenv()


class Settings:

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    LLM_MODEL = "gemma2-9b-it"

    SESSION_FOLDER = "sessions"

    LOG_FOLDER = "logs"


settings = Settings()