import os
from dotenv import load_dotenv

load_dotenv()

# Configuration variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-flash-latest"
