import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Directory and file paths
SESSIONS_DIR = "sessions"
CONTEXTS_FILE = "context.json"

# Default context configuration
DEFAULT_CONTEXT = {
    "name": "General Assistant",
    "instructions": "You are a helpful AI assistant. Respond in the language you are asked."
}

# Model configuration
MODEL_CONFIG = {
    "temperature": 0.7,
    "model": "deepseek-chat",
    "api_key": os.getenv("DEEPSEEK_API_KEY")
}
