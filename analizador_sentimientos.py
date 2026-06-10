from langchain_core.runnables import RunnableLambda, RunnableParallel
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

summary_branch = RunnableLambda(generate_summary)

#analisis de sentimientos en formato JSON
def analyze_sentiment(text):
    """Analiza el sentimiento y devuelve el resultado estructurado"""
    prompt = f"""Analiza el sentimiento del siguiente texto.
    Responde unicamente en formato JSON valido:
    {{"sentimiento": "positivo|negativo|neutro", "razon" : "justificacion breve"}}

    Texto: {text}"""

    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        return {"sentimiento": "neutro", "razon": "Error de analisis"}
    

sentiment_branch = RunnableLambda(analyze_sentiment)
    
#Combinacion de resulados 
def merge_results(data):
    """combina los resultados de ambas ramas en un formato unificado"""
    return {
        "resumen": data["resumen"],
        "sentimiento": data["sentimiento_data"]["sentimiento"],
        "razon" : data["sentimiento_data"]["razon"]

 }

merger = RunnableLambda(merge_results)


parallel_analysis = RunnableParallel({
    "resumen": summary_branch,
    "sentimiento_data": sentiment_branch
})

#cadena completa 
chain = preprocessor | parallel_analysis | merger 

review_batch = [
    "este producto es muy malo. no me ha gustado nada."
    "terrible calidad, no lo recomiendo"
    "esta bien, cumple su funcion"
]

resultado_batch = chain.batch
