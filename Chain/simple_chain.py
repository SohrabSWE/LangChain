from dotenv import load_dotenv
from langchain_groq import ChatGroq 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


prompt = PromptTemplate(
    template = """
    You are an AI Tutor. 
    Explain The following topic to a student in a simple and beginner-friendly way: 
    topic {topic}
         """, 
    input_variables=["topic"]
)


model = ChatGroq(model= "llama-3.3-70b-versatile")

parser = StrOutputParser()


chain = prompt | model | parser


result = chain.invoke({"topic":"Machine leanring"})


print(result)

chain.get_graph().print_ascii()