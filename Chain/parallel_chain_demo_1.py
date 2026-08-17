from dotenv import load_dotenv
from langchain_groq import ChatGroq 
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


load_dotenv()

model = ChatGroq(model= "llama-3.3-70b-versatile")

parser = StrOutputParser()


prompt1 = PromptTemplate(
    template = "Summarize the following text in 3 short lines: \n {text}",
    input_variables=["text"]
) 
prompt2 = PromptTemplate(
    template = "Write 3 simple quesions from the following text : \n {text}",
    input_variables=["text"]
) 


parallel_chain = RunnableParallel(
    {
        "Summary" : prompt1 | model | parser, 
        "questions": prompt2 | model | parser
    }
)

text = """
Deep learning is a subset of machine learning powered by multilayered artificial neural networks that mimic the human brain to process data and recognize complex patterns. It powers advanced technologies like computer vision, speech recognition, and generative AI.How Deep Learning WorksNeural Networks: Built using layers of connected artificial nodes or "neurons".Layers: Consists of an input layer, multiple hidden layers (the "deep" part), and an output layer.Feature Learning: Automatically extracts features from raw, unstructured data without manual help.Training: Adjusts internal weights and learns from mistakes over time using massive datasets
"""


result = parallel_chain.invoke({"text":text})


print("summary:\n", result["Summary"])
print("\nquestions:\n", result["questions"])


parallel_chain.get_graph().print_ascii()