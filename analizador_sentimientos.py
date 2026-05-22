from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
import json

# Configuración del modelo 
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

#procesador: limpia espacios y limita a 500 caracteres 
def preprocess_text(text):
    """"limpia el texto eliminando espacios extras y limitando longitud"""
    return text.strip()[:500]

preprocessor = RunnableLambda(preprocess_text)

#Generación de resumen 
def generate_summary(text):
    """Genera un resumen conciso del texto"""
    prompt = "Resume en una sola oración: {text}"
    response = llm.invoke(prompt)
    return response.content
