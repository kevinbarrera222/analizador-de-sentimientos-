"""App web del analizador de sentimientos (Streamlit).

Usa la misma cadena de LangChain que analizador_sentimientos.py:
preprocesamiento → (resumen ∥ sentimiento) → unión.
"""

import io
import os
import uuid
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from limite_uso import ZONA_HORARIA, LimitadorUso

CARPETA_ASSETS = Path(__file__).parent / "assets"
LOGO = str(CARPETA_ASSETS / "logo.svg")
ICONO = str(CARPETA_ASSETS / "icono.svg")

st.set_page_config(page_title="Analizador de Sentimientos", page_icon=ICONO, layout="centered")


def leer_config(nombre: str, por_defecto=None):
    """Lee un valor de los Secrets de Streamlit y, si no está, de las variables de entorno."""
    try:
        if nombre in st.secrets:
            return st.secrets[nombre]
    except Exception:
        pass
    return os.environ.get(nombre, por_defecto)


# La cadena lee la clave y el modelo desde variables de entorno
for variable in ("OPENAI_API_KEY", "OPENAI_MODEL"):
    valor = leer_config(variable)
    if valor:
        os.environ[variable] = str(valor)

from analizador_sentimientos import MAX_CARACTERES, MODELO, crear_cadena, crear_modelo  # noqa: E402

MAX_RESENAS_POR_ANALISIS = int(leer_config("MAX_RESENAS_POR_ANALISIS", 20))
MAX_RESENAS_POR_USUARIO = int(leer_config("MAX_RESENAS_POR_USUARIO", 60))
MAX_RESENAS_GLOBALES_DIA = int(leer_config("MAX_RESENAS_GLOBALES_DIA", 1500))
MAX_MB_CSV = 5

# Colores por sentimiento: polos azul (positivo) / rojo (negativo) con gris neutro
# en el medio. Validados contra el fondo oscuro de la app; además cada barra
# lleva su etiqueta y un emoji, así que el color nunca es la única pista.
SENTIMIENTOS = {
    "positivo": {"emoji": "😊", "color": "#3987e5", "etiqueta": "Positivo", "insignia": "blue"},
    "neutro": {"emoji": "😐", "color": "#8a8983", "etiqueta": "Neutro", "insignia": "gray"},
    "negativo": {"emoji": "😠", "color": "#e5484d", "etiqueta": "Negativo", "insignia": "red"},
    "error": {"emoji": "⚠️", "color": "#6b6a65", "etiqueta": "Error", "insignia": "orange"},
}

EJEMPLOS = """Este producto es muy malo. No me ha gustado nada.
Terrible calidad, no lo recomiendo.
Está bien, cumple su función.
¡Me encantó! Llegó rápido y funciona perfecto.
El servicio al cliente fue amable, pero el envío tardó dos semanas.
Excelente relación calidad-precio, lo volvería a comprar."""


@st.cache_resource
def obtener_limitador() -> LimitadorUso:
    return LimitadorUso(MAX_RESENAS_POR_USUARIO, MAX_RESENAS_GLOBALES_DIA)


@st.cache_resource
def obtener_cadena():
    return crear_cadena(crear_modelo())


def identificar_usuario() -> str:
    """Identifica al visitante por su IP; si no se puede, por su sesión."""
    headers = st.context.headers or {}
    reenviada = headers.get("X-Forwarded-For")
    if reenviada:
        return "ip:" + reenviada.split(",")[0].strip()
    ip = st.context.ip_address
    if isinstance(ip, str) and ip:
        return "ip:" + ip
    if "id_sesion" not in st.session_state:
        st.session_state.id_sesion = str(uuid.uuid4())
    return "sesion:" + st.session_state.id_sesion


def leer_csv(archivo) -> pd.DataFrame:
    """Lee un CSV detectando el separador (, ; tab) y la codificación."""
    datos = archivo.getvalue()
    for codificacion in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(datos), sep=None, engine="python", encoding=codificacion)
        except UnicodeDecodeError:
            continue
    raise ValueError("No pude leer el archivo. Guárdalo como CSV en UTF-8.")


def analizar(resenas: list[str]) -> list[dict]:
    """Analiza las reseñas en paralelo mostrando una barra de progreso."""
    cadena = obtener_cadena()
    resultados: list[dict | None] = [None] * len(resenas)
    progreso = st.progress(0.0, text="Analizando reseñas…")
    for completadas, (indice, salida) in enumerate(
        cadena.batch_as_completed(resenas, config={"max_concurrency": 5}, return_exceptions=True),
        start=1,
    ):
        if isinstance(salida, Exception):
            salida = {"resumen": "", "sentimiento": "error", "razon": f"No se pudo analizar: {salida}"}
        resultados[indice] = salida
        progreso.progress(completadas / len(resenas), text=f"Analizando reseñas… {completadas}/{len(resenas)}")
    progreso.empty()
    return resultados


def grafico_sentimientos(conteo: pd.DataFrame) -> alt.Chart:
    """Barras horizontales: una por sentimiento, con su cantidad y porcentaje al final."""
    orden = [s["etiqueta"] for s in SENTIMIENTOS.values()]
    colores = alt.Scale(
        domain=orden, range=[s["color"] for s in SENTIMIENTOS.values()]
    )
    base = alt.Chart(conteo).encode(
        y=alt.Y("Sentimiento:N", sort=orden, title=None, axis=alt.Axis(labelFontSize=13, ticks=False, domain=False)),
        x=alt.X("Reseñas:Q", title=None, axis=None, scale=alt.Scale(domainMax=float(conteo["Reseñas"].max()) * 1.25)),
        tooltip=[
            alt.Tooltip("Sentimiento:N"),
            alt.Tooltip("Reseñas:Q"),
            alt.Tooltip("Porcentaje:Q", format=".0%"),
        ],
    )
    barras = base.mark_bar(cornerRadiusEnd=4, height=22).encode(
        color=alt.Color("Sentimiento:N", scale=colores, legend=None)
    )
    etiquetas = base.mark_text(align="left", dx=8, fontSize=13, color="#C9CAD6").encode(text="Texto:N")
    return (barras + etiquetas).properties(height=alt.Step(40)).configure_view(stroke=None)


# -----------------------------
# ESTADO
# -----------------------------
limitador = obtener_limitador()
usuario = identificar_usuario()
st.session_state.setdefault("resultados", None)
st.session_state.setdefault("texto_resenas", "")

st.logo(LOGO, icon_image=ICONO, size="large")

# -----------------------------
# BARRA LATERAL
# -----------------------------
with st.sidebar:
    st.subheader("¿Cómo funciona?")
    st.markdown(
        "Por cada reseña, una cadena de **LangChain** ejecuta **en paralelo**:\n"
        "1. 📝 un **resumen** en una oración\n"
        "2. 💬 el **sentimiento** (positivo, neutro o negativo) con su razón\n\n"
        f"Modelo: **{MODELO}**"
    )
    st.divider()
    restantes = limitador.restantes(usuario)
    st.caption(f":material/analytics: Te quedan **{restantes} de {MAX_RESENAS_POR_USUARIO}** reseñas por analizar hoy.")
    st.progress(restantes / MAX_RESENAS_POR_USUARIO if MAX_RESENAS_POR_USUARIO else 0)
    st.caption(f"Máximo {MAX_RESENAS_POR_ANALISIS} reseñas por análisis.")

if not os.environ.get("OPENAI_API_KEY"):
    st.error(
        "🔑 Falta la API Key de OpenAI. Agrega `OPENAI_API_KEY` en los Secrets de la app "
        "(en Streamlit Cloud: Manage app → ⋮ → Settings → Secrets)."
    )
    st.stop()

# -----------------------------
# ENTRADA
# -----------------------------
st.markdown("## Analizador de Sentimientos")
st.markdown(
    "Pega reseñas o comentarios de clientes y descubre en segundos qué opinan: "
    "el sentimiento de cada uno, por qué, y un resumen."
)

pestana_texto, pestana_csv = st.tabs([":material/edit_note: Escribir reseñas", ":material/upload_file: Subir CSV"])

resenas: list[str] = []
with pestana_texto:
    if st.button("Probar con ejemplos", icon=":material/auto_awesome:", type="tertiary"):
        st.session_state.texto_resenas = EJEMPLOS
    texto = st.text_area(
        "Una reseña por línea",
        key="texto_resenas",
        height=200,
        placeholder="Me encantó el producto, llegó rápido.\nNo funciona como esperaba…",
    )
    resenas_texto = [linea.strip() for linea in texto.splitlines() if linea.strip()]

with pestana_csv:
    archivo = st.file_uploader(
        "Archivo CSV con una columna de reseñas",
        type=["csv"],
        max_upload_size=MAX_MB_CSV,
    )
    resenas_csv: list[str] = []
    if archivo is not None:
        try:
            tabla = leer_csv(archivo)
        except Exception as e:
            st.warning(f"No pude leer el CSV: {e}")
        else:
            columnas_texto = [c for c in tabla.columns if not pd.api.types.is_numeric_dtype(tabla[c])] or list(tabla.columns)
            # Se sugiere la columna con los textos más largos (normalmente, la de las reseñas)
            columnas_texto.sort(key=lambda c: tabla[c].astype(str).str.len().mean(), reverse=True)
            columna = st.selectbox("¿Qué columna tiene las reseñas?", columnas_texto)
            resenas_csv = [str(v).strip() for v in tabla[columna].dropna() if str(v).strip()]
            st.caption(f"{len(resenas_csv)} reseñas encontradas en la columna «{columna}».")

# Si hay un CSV cargado se usa ese; si no, el texto
resenas = resenas_csv or resenas_texto
origen = "CSV" if resenas_csv else "texto"

if st.button(
    f"Analizar {len(resenas)} reseña{'s' if len(resenas) != 1 else ''}" if resenas else "Analizar",
    type="primary",
    icon=":material/play_arrow:",
    disabled=not resenas,
    width="stretch",
):
    if len(resenas) > MAX_RESENAS_POR_ANALISIS:
        st.info(f"Se analizarán las primeras {MAX_RESENAS_POR_ANALISIS} reseñas del {origen}.")
        resenas = resenas[:MAX_RESENAS_POR_ANALISIS]
    if not limitador.intentar_consumir(usuario, len(resenas)):
        disponibles = limitador.restantes(usuario)
        if disponibles:
            st.warning(f"Hoy solo te quedan {disponibles} reseñas por analizar. Reduce la lista e inténtalo de nuevo.")
        else:
            st.warning("Llegaste al límite de reseñas por hoy. Vuelve mañana. 🙌")
        st.stop()

    resultados = analizar(resenas)
    fallidas = sum(r["sentimiento"] == "error" for r in resultados)
    if fallidas:
        limitador.devolver(usuario, fallidas)  # lo que falló no se cobra
    st.session_state.resultados = pd.DataFrame(
        {
            "Reseña": [r[:MAX_CARACTERES] for r in resenas],
            "Sentimiento": [r["sentimiento"] for r in resultados],
            "Razón": [r["razon"] for r in resultados],
            "Resumen": [r["resumen"] for r in resultados],
        }
    )
    st.rerun()  # actualiza el contador de la barra lateral

# -----------------------------
# RESULTADOS
# -----------------------------
df = st.session_state.resultados
if df is not None and len(df):
    st.divider()
    st.markdown("### Resultados")

    errores = int((df["Sentimiento"] == "error").sum())
    if errores == len(df):
        st.error("No se pudo analizar ninguna reseña. Revisa tu API Key y que tu cuenta de OpenAI tenga saldo.")
        st.caption(df["Razón"].iloc[0])
        st.stop()
    if errores:
        st.warning(f"{errores} reseña(s) no se pudieron analizar y no se descontaron de tu límite.")

    total = len(df)
    columnas = st.columns(4)
    columnas[0].metric("Reseñas", total, border=True)
    for columna, clave in zip(columnas[1:], ["positivo", "neutro", "negativo"]):
        cantidad = int((df["Sentimiento"] == clave).sum())
        info = SENTIMIENTOS[clave]
        columna.metric(f"{info['emoji']} {info['etiqueta']}", f"{cantidad / total:.0%}", f"{cantidad} reseña{'s' if cantidad != 1 else ''}", delta_color="off", delta_arrow="off", border=True)

    conteo = (
        df["Sentimiento"].value_counts().rename_axis("clave").reset_index(name="Reseñas")
    )
    conteo["Sentimiento"] = conteo["clave"].map(lambda c: SENTIMIENTOS.get(c, SENTIMIENTOS["error"])["etiqueta"])
    conteo["Porcentaje"] = conteo["Reseñas"] / total
    conteo["Texto"] = conteo.apply(lambda f: f"{f['Reseñas']} · {f['Porcentaje']:.0%}", axis=1)
    st.altair_chart(grafico_sentimientos(conteo), width="stretch")

    st.markdown("#### Detalle por reseña")
    presentes = [c for c in SENTIMIENTOS if (df["Sentimiento"] == c).any()]
    filtro = st.pills(
        "Filtrar por sentimiento",
        presentes,
        selection_mode="multi",
        default=presentes,
        format_func=lambda c: f"{SENTIMIENTOS[c]['emoji']} {SENTIMIENTOS[c]['etiqueta']}",
        label_visibility="collapsed",
    )
    visibles = df[df["Sentimiento"].isin(filtro or [])]
    with st.container(height=520 if len(visibles) > 4 else "content", border=False):
        for _, fila in visibles.iterrows():
            info = SENTIMIENTOS.get(fila["Sentimiento"], SENTIMIENTOS["error"])
            with st.container(border=True):
                st.markdown(f":{info['insignia']}-badge[{info['emoji']} {info['etiqueta']}]")
                st.markdown(fila["Reseña"])
                st.caption(f"**Por qué:** {fila['Razón']}")
                if fila["Resumen"]:
                    st.caption(f"**Resumen:** {fila['Resumen']}")
    if not len(visibles):
        st.caption("Elige al menos un sentimiento para ver las reseñas.")

    st.download_button(
        "Descargar resultados (CSV)",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"sentimientos-{datetime.now(ZONA_HORARIA):%Y%m%d-%H%M}.csv",
        mime="text/csv",
        icon=":material/download:",
    )
