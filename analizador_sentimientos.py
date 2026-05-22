from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
import json

# Configuración del modelo 
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)