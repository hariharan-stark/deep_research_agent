import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
    RESEARCHER_MODEL = os.getenv("RESEARCHER_MODEL", "gemini-1.5-pro")
    ANALYST_MODEL = os.getenv("ANALYST_MODEL", "gemini-1.5-pro")
    
    @staticmethod
    def validate():
        missing = []
        if not Config.GOOGLE_APPLICATION_CREDENTIALS:
            missing.append("GOOGLE_APPLICATION_CREDENTIALS")
        if not Config.OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not Config.GOOGLE_API_KEY:
            missing.append("GOOGLE_API_KEY")
        if not Config.GOOGLE_CSE_ID:
            missing.append("GOOGLE_CSE_ID")
            
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")

config = Config()
