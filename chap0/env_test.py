import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일을 읽어서 os.environ 에 주입

key = os.getenv("OPENAI_API_KEY")
print("키 앞 10자:", key[:10] if key else "None")
print("키 길이:", len(key) if key else 0)