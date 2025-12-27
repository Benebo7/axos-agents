"""Utilitários de streaming para ferramentas do agente de análise."""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from langgraph.config import get_stream_writer


def get_stream_writer_safe():
    """Obtém o writer de streaming do LangGraph, tratando falhas silenciosamente."""
    try:
        writer = get_stream_writer()
        print("✅ StreamWriter obtido com sucesso")
        return writer
    except Exception as error:  # noqa: BLE001
        print(f"⚠️ Não foi possível obter StreamWriter: {error}")
        return None


def stream_custom_event(writer, event_type: str, data: Dict[str, Any]) -> bool:
    """Envia um evento customizado para o front se houver writer disponível."""
    if not writer:
        return False

    try:
        writer({
            "type": event_type,
            "data": data,
        })
        return True
    except Exception as error:  # noqa: BLE001
        print(f"⚠️ Erro ao fazer stream de '{event_type}': {error}")
        return False


def stream_progress_event(
    writer,
    *,
    stage: str,
    status: str,
    message: str,
    order: int,
    total_steps: int,
    details: Optional[Dict[str, Any]] = None,
) -> bool:
    """Convenience helper para emitir eventos de progresso padronizados."""
    payload: Dict[str, Any] = {
        "stage": stage,
        "status": status,  # pending | running | completed | error
        "message": message,
        "order": order,
        "total_steps": total_steps,
        "timestamp": time.time(),
    }

    if details:
        payload["details"] = details

    return stream_custom_event(writer, "progress", payload)

