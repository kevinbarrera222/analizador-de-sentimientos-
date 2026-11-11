"""Límite de reseñas analizadas por usuario y por día.

Los contadores viven en memoria del servidor (compartidos entre todas las
sesiones con ``st.cache_resource``) y se reinician cada día a la medianoche
de Colombia, o cuando la app se reinicia o se "duerme" en Streamlit Cloud.
"""

from __future__ import annotations

import threading
from datetime import datetime
from zoneinfo import ZoneInfo

ZONA_HORARIA = ZoneInfo("America/Bogota")


def _hoy() -> str:
    return datetime.now(ZONA_HORARIA).date().isoformat()


class LimitadorUso:
    """Cuenta cuántas reseñas analiza cada usuario por día.

    El tope global diario protege el saldo de OpenAI aunque alguien logre
    saltarse el límite individual (por ejemplo, cambiando de red).
    """

    def __init__(self, max_por_usuario: int, max_global: int) -> None:
        self.max_por_usuario = max_por_usuario
        self.max_global = max_global
        self._lock = threading.Lock()
        self._dia = _hoy()
        self._por_usuario: dict[str, int] = {}
        self._total = 0

    def _reiniciar_si_cambio_el_dia(self) -> None:
        if _hoy() != self._dia:
            self._dia = _hoy()
            self._por_usuario.clear()
            self._total = 0

    def restantes(self, usuario: str) -> int:
        with self._lock:
            self._reiniciar_si_cambio_el_dia()
            propio = self.max_por_usuario - self._por_usuario.get(usuario, 0)
            return max(0, min(propio, self.max_global - self._total))

    def intentar_consumir(self, usuario: str, cantidad: int) -> bool:
        """Descuenta ``cantidad`` reseñas si alcanzan; si no, no descuenta nada."""
        with self._lock:
            self._reiniciar_si_cambio_el_dia()
            usados = self._por_usuario.get(usuario, 0)
            if usados + cantidad > self.max_por_usuario or self._total + cantidad > self.max_global:
                return False
            self._por_usuario[usuario] = usados + cantidad
            self._total += cantidad
            return True

    def devolver(self, usuario: str, cantidad: int) -> None:
        """Reintegra reseñas que no se pudieron analizar por un error de OpenAI."""
        with self._lock:
            cantidad = min(cantidad, self._por_usuario.get(usuario, 0))
            self._por_usuario[usuario] = self._por_usuario.get(usuario, 0) - cantidad
            self._total = max(0, self._total - cantidad)
