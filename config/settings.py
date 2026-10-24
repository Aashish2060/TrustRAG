from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    data_dir: str = os.getenv("TRUSTRAG_DATA_DIR", "data")
    chunk_size: int = int(os.getenv("TRUSTRAG_CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("TRUSTRAG_CHUNK_OVERLAP", "100"))
    top_k: int = int(os.getenv("TRUSTRAG_TOP_K", "4"))
    max_corrections: int = int(os.getenv("TRUSTRAG_MAX_CORRECTIONS", "2"))
    min_relevance: float = float(os.getenv("TRUSTRAG_MIN_RELEVANCE", "0.05"))
    llm_base_url: str = os.getenv("LLM_BASE_URL", "").strip()
    llm_api_key: str = os.getenv("LLM_API_KEY", "").strip()
    llm_model: str = os.getenv("LLM_MODEL", "").strip()

settings = Settings()
