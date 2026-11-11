# 🧠 Analizador de Sentimientos con LangChain y OpenAI

Este proyecto implementa un pipeline completo de análisis de texto utilizando **LangChain**, **OpenAI** y procesamiento funcional.
El sistema genera un **resumen** del texto y un **análisis de sentimiento**, ejecutados en paralelo para mayor eficiencia.

---

## 🚀 Características principales

- 🔥 **Preprocesamiento automático** del texto (quita espacios extra y limita a 500 caracteres sin cortar palabras)
- 🧠 **Generación de resumen** usando LLM
- 💬 **Análisis de sentimiento** (positivo, negativo o neutro) con **salida estructurada** (`with_structured_output` + Pydantic): el modelo siempre responde con el formato correcto
- ⚡ **Ejecución paralela** con `RunnableParallel`
- 🔗 **Pipeline completo** usando operadores de LangChain (`|`)
- 📦 **Procesamiento por lotes (batch)** para múltiples textos
- ⚠️ Los errores se reportan como `error`, no se disfrazan de "neutro"

---

## 🔗 Cómo funciona la cadena

```
texto → preprocesamiento → ┬→ resumen ─────┬→ unión → {resumen, sentimiento, razon}
                           └→ sentimiento ─┘
```

---

## 💻 Cómo ejecutarlo

Necesitas Python 3.10 o superior y una API Key de OpenAI.

```bash
pip install -r requirements.txt
```

Configura tu clave:

```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."

# Mac / Linux
export OPENAI_API_KEY="sk-..."
```

Ejecuta:

```bash
python analizador_sentimientos.py
```

Ejemplo de salida (la redacción exacta de la razón y el resumen cambia en cada ejecución):

```
Analizando 4 reseñas con gpt-6-luna...

[1] Este producto es muy malo. No me ha gustado nada.
    Sentimiento: 😠 NEGATIVO
    Razón:       El usuario expresa una opinión claramente desfavorable del producto.
    Resumen:     El usuario está muy insatisfecho con el producto.
```

Para analizar tus propios textos, edita la lista `review_batch` en `analizador_sentimientos.py`.

**Modelo:** por defecto usa `gpt-6-luna`, el modelo más económico de OpenAI (reemplaza a `gpt-4o-mini`, que OpenAI retira de su API el 14 de enero de 2027). Puedes usar otro con la variable de entorno `OPENAI_MODEL`.

---

## 📁 Estructura del proyecto

```
analizador_sentimientos.py   # Cadena de LangChain, ejemplos y ejecución
requirements.txt             # Librerías con versiones fijas
README.md
```

---

## 🏗️ Tecnologías

- Python 3.10+
- LangChain (LCEL, `RunnableLambda`, `RunnableParallel`, `ChatPromptTemplate`)
- OpenAI API (GPT-6 Luna)
- Pydantic (salida estructurada)
