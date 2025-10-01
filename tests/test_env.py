from dotenv import load_dotenv
import os

load_dotenv()
print("DATABASE_URL:", os.getenv("DATABASE_URL"))
print("REDIS_URL:", os.getenv("REDIS_URL"))
print("Environment loaded successfully!")