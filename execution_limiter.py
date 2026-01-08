"""
Limitador de execuções simultâneas para os agents.
Usa Semaphore para controlar quantas análises rodam em paralelo.

Suporta tanto código sync (threading.Semaphore) quanto async (asyncio.Semaphore).
"""
import asyncio
import threading
import os
from functools import wraps
from contextlib import contextmanager

# Configurável via env var, padrão: 2 execuções simultâneas
MAX_CONCURRENT_EXECUTIONS = int(os.getenv("MAX_CONCURRENT_EXECUTIONS", "2"))

# Semaphores globais
_async_semaphore = asyncio.Semaphore(MAX_CONCURRENT_EXECUTIONS)
_sync_semaphore = threading.Semaphore(MAX_CONCURRENT_EXECUTIONS)

# Contadores para debug/monitoramento
_active_async = 0
_active_sync = 0
_lock = threading.Lock()


def get_stats() -> dict:
    """Retorna estatísticas de execuções ativas."""
    return {
        "max_concurrent": MAX_CONCURRENT_EXECUTIONS,
        "active_async": _active_async,
        "active_sync": _active_sync,
        "total_active": _active_async + _active_sync,
    }


# ============================================
# SYNC (para tools do LangChain)
# ============================================

@contextmanager
def sync_limit():
    """
    Context manager sync para limitar execuções.

    Uso:
        with sync_limit():
            # código pesado aqui
    """
    global _active_sync
    _sync_semaphore.acquire()
    with _lock:
        _active_sync += 1
        print(f"🔒 [SYNC] Semaphore acquired. Active: {_active_sync}/{MAX_CONCURRENT_EXECUTIONS}")
    try:
        yield
    finally:
        _sync_semaphore.release()
        with _lock:
            _active_sync -= 1
            print(f"🔓 [SYNC] Semaphore released. Active: {_active_sync}/{MAX_CONCURRENT_EXECUTIONS}")


def limited_sync(func):
    """
    Decorator para limitar execuções de funções sync.

    Uso:
        @limited_sync
        def my_heavy_function():
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        with sync_limit():
            return func(*args, **kwargs)
    return wrapper


# ============================================
# ASYNC (para código async)
# ============================================

class ExecutionLimiter:
    """
    Async context manager para limitar execuções.

    Uso:
        async with ExecutionLimiter():
            await graph.ainvoke(state)
    """

    async def __aenter__(self):
        global _active_async
        await _async_semaphore.acquire()
        with _lock:
            _active_async += 1
            print(f"🔒 [ASYNC] Semaphore acquired. Active: {_active_async}/{MAX_CONCURRENT_EXECUTIONS}")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        global _active_async
        _async_semaphore.release()
        with _lock:
            _active_async -= 1
            print(f"🔓 [ASYNC] Semaphore released. Active: {_active_async}/{MAX_CONCURRENT_EXECUTIONS}")
        return False


def limited_async(func):
    """
    Decorator para limitar execuções de funções async.

    Uso:
        @limited_async
        async def my_heavy_function():
            ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        async with ExecutionLimiter():
            return await func(*args, **kwargs)
    return wrapper
