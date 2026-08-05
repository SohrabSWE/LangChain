from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv 

load_dotenv()

llm = GoogleGenerativeAI(
    model = "gemini-3.6-flash",
)

result = llm.invoke("What is the Capital of Bangladesh")

print(result)