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


#step 1 : invoke the prompt 

prompt_value = prompt.invoke({"topic":"Machine Learning"})


# print(prompt_value)


#step 2 : send the prompt to the model 

model_output = model.invoke(prompt_value)

# print(model_output)

#step 3: parse the model's response 

final_output = parser.invoke(model_output)

print(final_output)