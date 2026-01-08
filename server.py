"""
Servidor principal que expõe os grafos LangGraph com controle de concorrência.

Este servidor substitui o `langgraph dev` e adiciona:
- Controle de concorrência global (semaphore)
- Endpoint /health
- Autenticação via API key

Uso:
    python server.py
    # ou
    uvicorn server:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import json
import os
from typing import AsyncIterator

from execution_limiter import ExecutionLimiter, get_stats

# ============================================
# AUTH
# ============================================

API_KEY = os.getenv("API_KEY", "")
security = HTTPBearer(auto_error=False)


async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verifica API key no header Authorization: Bearer <key>
    Se API_KEY não estiver configurada, permite acesso (dev mode).
    """
    if not API_KEY:
        # Dev mode - sem autenticação
        return True

    if not credentials:
        raise HTTPException(status_code=401, detail="Missing API key")

    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return True

# Importar os grafos
from agent import graph as crypto_analysis_graph
from chat_agent import chat_graph
from news_agent import news_graph
from report_agent import report_graph
from research_agent import research_graph

app = FastAPI(title="Axos Analysis Agents", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mapeamento de agent_id para grafo
GRAPHS = {
    "crypto_analysis_agent": crypto_analysis_graph,
    "chat_agent": chat_graph,
    "news_agent": news_graph,
    "report_agent": report_graph,
    "research_agent": research_graph,
}


@app.get("/health")
def health_check():
    """Health check + status do semaphore de concorrência."""
    stats = get_stats()
    return {
        "status": "healthy",
        "concurrency": stats,
        "available_agents": list(GRAPHS.keys()),
    }


@app.get("/agents", dependencies=[Depends(verify_api_key)])
def list_agents():
    """Lista todos os agents disponíveis."""
    return {"agents": list(GRAPHS.keys())}


@app.post("/runs", dependencies=[Depends(verify_api_key)])
async def create_run(request: Request):
    """
    Executa um agent com controle de concorrência (sem streaming).

    Body:
    {
        "assistant_id": "crypto_analysis_agent",  // ou "agent_id"
        "input": { ... },
        "config": { ... }
    }
    """
    body = await request.json()
    # Suporta tanto assistant_id (BFF) quanto agent_id
    agent_id = body.get("assistant_id") or body.get("agent_id")
    input_data = body.get("input", {})
    config = body.get("config", {})

    if agent_id not in GRAPHS:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_id}' not found. Available: {list(GRAPHS.keys())}"
        )

    graph = GRAPHS[agent_id]

    # Controle de concorrência
    async with ExecutionLimiter():
        try:
            result = await graph.ainvoke(input_data, config=config)
            return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}


@app.post("/runs/stream", dependencies=[Depends(verify_api_key)])
async def create_run_stream(request: Request):
    """
    Executa um agent com streaming e controle de concorrência.
    Compatível com o formato esperado pelo BFF (Frapto).

    Body:
    {
        "assistant_id": "crypto_analysis_agent",  // ou "agent_id"
        "input": { ... },
        "config": { ... },
        "stream_mode": ["custom", "values"] ou "messages"
    }

    Response SSE:
        event: custom
        data: {"type": "progress", "data": {...}}

        event: values
        data: {"opportunities": [...], ...}

        event: messages
        data: ["ai", {"content": "...", ...}]  // para stream_mode: "messages"
    """
    body = await request.json()
    # Suporta tanto assistant_id (BFF) quanto agent_id
    agent_id = body.get("assistant_id") or body.get("agent_id")
    input_data = body.get("input", {})
    config = body.get("config", {})
    stream_mode_raw = body.get("stream_mode", ["custom", "values"])

    # Normalizar stream_mode para lista (BFF pode enviar string ou lista)
    if isinstance(stream_mode_raw, str):
        stream_mode = [stream_mode_raw]
    else:
        stream_mode = stream_mode_raw

    if agent_id not in GRAPHS:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_id}' not found. Available: {list(GRAPHS.keys())}"
        )

    graph = GRAPHS[agent_id]

    async def generate() -> AsyncIterator[str]:
        # Controle de concorrência
        async with ExecutionLimiter():
            try:
                async for event_type, chunk in graph.astream(input_data, config=config, stream_mode=stream_mode):
                    # Formato SSE compatível com langgraph-stream.js
                    # event: <type>\ndata: <json>\n\n

                    # Para stream_mode "messages", o BFF espera:
                    # data: [message_type, message_data]
                    # onde message_data.content contém o texto
                    if event_type == "messages":
                        # chunk já é o objeto de mensagem, empacotar no formato esperado
                        # Formato: [type, {content: "...", ...}]
                        msg_type = chunk.get("type", "ai") if isinstance(chunk, dict) else "ai"
                        formatted_data = [msg_type, chunk]
                        yield f"event: {event_type}\n"
                        yield f"data: {json.dumps(formatted_data)}\n\n"
                    else:
                        yield f"event: {event_type}\n"
                        yield f"data: {json.dumps(chunk)}\n\n"
            except Exception as e:
                yield f"event: error\n"
                yield f"data: {json.dumps({'message': str(e)})}\n\n"
            finally:
                yield f"event: end\n"
                yield f"data: {{}}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ============================================
# LIFECYCLE
# ============================================

@app.on_event("startup")
def startup_event():
    print("🚀 Server started with concurrency control")
    print(f"   MAX_CONCURRENT_EXECUTIONS={os.getenv('MAX_CONCURRENT_EXECUTIONS', '2')}")
    if API_KEY:
        print("   AUTH: API key required")
    else:
        print("   AUTH: disabled (dev mode)")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
