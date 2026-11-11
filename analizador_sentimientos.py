"""Analizador de sentimientos con LangChain y OpenAI.

Para cada texto, la cadena:
1. limpia el texto (preprocesamiento),
2. genera un resumen y analiza el sentimiento EN PARALELO,
3. une ambos resultados en un solo diccionario.
"""

import os
import sys
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# gpt-6-luna: el modelo más económico de OpenAI. Reemplaza a gpt-4o-mini, que
# OpenAI retira de su API el 14 de enero de 2027. Se puede cambiar con la
# variable de entorno OPENAI_MODEL.
MODELO = os.environ.get("OPENAI_MODEL", "gpt-6-luna")
MAX_CARACTERES = 500


# -----------------------------
# 1. PREPROCESAMIENTO
# -----------------------------
def preprocess_text(text: str) -> str:
    """Limpia el texto: quita espacios extra y lo limita a MAX_CARACTERES sin cortar palabras."""
    limpio = " ".join(text.split())  # también une saltos de línea y espacios dobles
    if len(limpio) <= MAX_CARACTERES:
        return limpio
    recortado = limpio[:MAX_CARACTERES]
    return recortado.rsplit(" ", 1)[0] if " " in recortado else recortado


# -----------------------------
# 2. ANÁLISIS (resumen + sentimiento)
# -----------------------------
class AnalisisSentimiento(BaseModel):
    """Formato exacto que debe devolver el modelo."""

    sentimiento: Literal["positivo", "negativo", "neutro"] = Field(
        description="Sentimiento general del texto"
    )
    razon: str = Field(description="Justificación breve, en una frase")


prompt_resumen = ChatPromptTemplate.from_template(
    "Resume en una sola oración el siguiente texto:\n\n{texto}"
)

prompt_sentimiento = ChatPromptTemplate.from_template(
    "Analiza el sentimiento del siguiente texto y explica brevemente por qué.\n\n"
    "Texto: {texto}"
)


def crear_modelo() -> ChatOpenAI:
    if MODELO.startswith(("gpt-5", "gpt-6", "o")):
        # Modelos de razonamiento: no admiten temperatura; "none" = responde sin pensar de más
        return ChatOpenAI(model=MODELO, reasoning_effort="none")
    return ChatOpenAI(model=MODELO, temperature=0)


def crear_cadena(llm):
    """Arma la cadena completa: preprocesamiento → (resumen ∥ sentimiento) → unión."""
    preprocessor = RunnableLambda(preprocess_text)

    # Los prompts reciben {"texto": ...}
    a_variables = RunnableLambda(lambda texto: {"texto": texto})

    summary_branch = a_variables | prompt_resumen | llm | RunnableLambda(lambda r: r.content.strip())

    # with_structured_output obliga al modelo a responder con el formato de
    # AnalisisSentimiento: no hay que parsear JSON a mano.
    analizador = prompt_sentimiento | llm.with_structured_output(AnalisisSentimiento)

    def analyze_sentiment(texto: str) -> dict:
        try:
            return analizador.invoke({"texto": texto}).model_dump()
        except Exception as e:
            # Un error se informa como error, no se disfraza de "neutro"
            return {"sentimiento": "error", "razon": f"No se pudo analizar: {e}"}

    sentiment_branch = RunnableLambda(analyze_sentiment)

    parallel_analysis = RunnableParallel(
        {
            "resumen": summary_branch,
            "sentimiento_data": sentiment_branch,
        }
    )

    def merge_results(data: dict) -> dict:
        """Combina los resultados de ambas ramas en un formato unificado."""
        return {
            "resumen": data["resumen"],
            "sentimiento": data["sentimiento_data"]["sentimiento"],
            "razon": data["sentimiento_data"]["razon"],
        }

    merger = RunnableLambda(merge_results)

    return preprocessor | parallel_analysis | merger


# -----------------------------
# 3. EJEMPLO DE USO
# -----------------------------
review_batch = [
    "Este producto es muy malo. No me ha gustado nada.",
    "Terrible calidad, no lo recomiendo.",
    "Está bien, cumple su función.",
    "¡Me encantó! Llegó rápido y funciona perfecto.",
]

EMOJIS = {"positivo": "😊", "negativo": "😠", "neutro": "😐", "error": "⚠️"}


def mostrar_resultados(textos: list[str], resultados: list[dict]) -> None:
    for i, (texto, r) in enumerate(zip(textos, resultados), start=1):
        emoji = EMOJIS.get(r["sentimiento"], "")
        print(f"\n[{i}] {texto}")
        print(f"    Sentimiento: {emoji} {r['sentimiento'].upper()}")
        print(f"    Razón:       {r['razon']}")
        print(f"    Resumen:     {r['resumen']}")


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit(
            "Falta la API Key de OpenAI.\n"
            "Configúrala antes de ejecutar, por ejemplo:\n"
            "  Windows (PowerShell):  $env:OPENAI_API_KEY=\"sk-...\"\n"
            "  Mac / Linux:           export OPENAI_API_KEY=\"sk-...\""
        )

    chain = crear_cadena(crear_modelo())
    print(f"Analizando {len(review_batch)} reseñas con {MODELO}...")
    try:
        resultados = chain.batch(review_batch)
    except Exception as e:
        sys.exit(f"Error al llamar a OpenAI: {e}\nRevisa tu API Key y que tu cuenta tenga saldo.")
    mostrar_resultados(review_batch, resultados)


if __name__ == "__main__":
    main()
