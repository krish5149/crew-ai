from dotenv import load_dotenv
from pathlib import Path
import os



load_dotenv()

print("OPENROUTER_API_KEY =", os.getenv("SERPER_API_KEY"))