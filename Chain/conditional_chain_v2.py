from dotenv import load_dotenv
from langchain_groq import ChatGroq 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

load_dotenv()

model = ChatGroq(model= "llama-3.3-70b-versatile")


parser = StrOutputParser()

# step1: classify the review
classifier_prompt = PromptTemplate(
    template = """
    You are a movie review classifier.
    Classify the following review as either:
    - positive
    - negative
    Return only one word: Positive or Negative.
    Review : {review}
""",
    input_variables = ["review"]
)

classifier_chain = classifier_prompt | model | parser

positive_prompt = PromptTemplate(
    template= """
    Reply to this positive movie review in a friendly way.
    Review : {review}
""",
input_variables= ["review"]
)

negative_prompt = PromptTemplate(
    template= """
    Reply to this negative movie review by apolizing and offering help 
    Review : {review}
""",
input_variables= ["review"]
)

positive_chain = positive_prompt | model | parser
negative_chain = negative_prompt | model | parser

review = "The movie was absolutely fantastic. I love it and enjoy every single moment."

sentiment = classifier_chain.invoke({"review": review})

print("Predicted sentiment", sentiment)

conditional_chain = RunnableBranch(
    (
        lambda x : x["sentiment"].strip().lower() == "positive", positive_chain
    ),
    negative_chain
)

result = conditional_chain.invoke({
    "review" : review,
    "sentiment" : sentiment
})


print(result)