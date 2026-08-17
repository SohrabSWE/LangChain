from dotenv import load_dotenv
from langchain_groq import ChatGroq 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

load_dotenv()

model = ChatGroq(model= "qwen/qwen3.6-27b")


parser = StrOutputParser()

positive_prompt = ChatPromptTemplate.from_template(
    "Reply to this positive movie review in a friendly way: \n {review}"
)

negative_prompt = ChatPromptTemplate.from_template(
    "Reply to this negative movie review by apolizing and offering help: \n {review}"
)

positive_chain = positive_prompt | model | parser
negative_chain = negative_prompt | model | parser

conditional_chain = RunnableBranch(
    (
        lambda x : "good" in x["review"].lower(), positive_chain
    ),
    negative_chain
)

result = conditional_chain.invoke({
    "review" : "The movie was really good and I enjoyed every moment."
})


print(result)