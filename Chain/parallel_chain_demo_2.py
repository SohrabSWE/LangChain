from dotenv import load_dotenv
from langchain_groq import ChatGroq 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


load_dotenv()

model1 = ChatGroq(model= "llama-3.3-70b-versatile")
model2 = ChatGroq(model= "qwen/qwen3.6-27b")




prompt1 = PromptTemplate(
    template = "Gnerate short and simple notes from the following text \n {text}",
    input_variables=["text"]
) 
prompt2 = PromptTemplate(
    template = "Generate 5 short question from the following text \n {text}",
    input_variables=["text"]
) 

prompt3 = PromptTemplate(
    template = "Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=["notes","quiz"]
) 

parser = StrOutputParser()


parallel_chain = RunnableParallel(
    {
        "notes" : prompt1 | model1 | parser, 
        "quiz": prompt2 | model2 | parser
    }
)


merge_chain = prompt3 | model1 | parser #workflow 

chain = parallel_chain | merge_chain

text = """
Deep learning is a subset of machine learning powered by multilayered artificial neural networks that mimic the human brain to process data and recognize complex patterns. It powers advanced technologies like computer vision, speech recognition, and generative AI.How Deep Learning WorksNeural Networks: Built using layers of connected artificial nodes or "neurons".Layers: Consists of an input layer, multiple hidden layers (the "deep" part), and an output layer.Feature Learning: Automatically extracts features from raw, unstructured data without manual help.Training: Adjusts internal weights and learns from mistakes over time using massive datasets
"""


result = chain.invoke({"text":text})


print(result)


chain.get_graph().print_ascii()