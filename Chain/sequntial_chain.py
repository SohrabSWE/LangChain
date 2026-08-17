from dotenv import load_dotenv
from langchain_groq import ChatGroq 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


prompt1 = PromptTemplate(
    template = """
    Evaluate the following Student's Answer. 
    Question : {question}
    Student's Answer : {answer}

    Provide a detailed evaluation inlcuding: 
    -correctness 
    -strengths 
    -weaknesses
    -suggestions for improvement

         """, 
    input_variables=["question","answer"]
)

prompt2 = PromptTemplate(
    template = """
    Convert the following detailed evaluation into a concise 5-point feedback:
    {evaluation}

         """, 
    input_variables=["evaluation"]
)


model = ChatGroq(model= "llama-3.3-70b-versatile")

parser = StrOutputParser()

#chain 
chain = prompt1 | model | parser | prompt2 | model | parser


result = chain.invoke({
    "question":"What is Machine leanring",
    "answer": """
   Machine learning is a subset of artificial intelligence where computers learn patterns from data and make predictions without being explicitly programmed for every step 
         """
    })


print(result)

chain.get_graph().print_ascii()