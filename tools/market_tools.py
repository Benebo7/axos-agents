"""Ferramentas para buscar dados de mercado"""
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from typing import List, Dict, Annotated
import httpx
import os
from tools.stream_utils import get_stream_writer_safe, stream_progress_event

# CoinGecko Pro API configuration
COINGECKO_PRO_API_KEY = os.getenv("COINGECKO_PRO_API_KEY") or os.getenv("VITE_COINGECKO_PRO_API_KEY")
COINGECKO_BASE_URL = (
    "https://pro-api.coingecko.com/api/v3" if COINGECKO_PRO_API_KEY 
    else "https://api.coingecko.com/api/v3"
)

@tool
def fetch_market_data(
    period: str = "7d", 
    limit: int = 20,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """
    Busca dados atualizados do mercado de criptomoedas via CoinGecko
    
    **NOTE**: This tool uses CoinGecko API directly. For better rate limit management,
    consider routing through the frontend API proxy which has centralized caching
    and request deduplication.

    **IMPORTANTE**: Esta tool salva os dados no estado 'market_data'.
    NÃO retorna os dados brutos para o LLM.

    Args:
        period: Período para análise (1d, 7d, 30d, 90d)
        limit: Número de moedas para retornar (recomendado: 20-50 para evitar rate limits)

    Returns:
        Command: Atualiza state['market_data'] e retorna mensagem de sucesso.
    """
    writer = get_stream_writer_safe()
    stream_progress_event(
        writer,
        stage="fetch_market_data",
        status="running",
        message="Coletando dados do mercado (CoinGecko)",
        order=1,
        total_steps=4,
        details={"period": period, "limit": limit},
    )

    api_url = f"{COINGECKO_BASE_URL}/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": limit,
        "page": 1,
        "sparkline": False,
        "price_change_percentage": "24h"
    }

    # Add Pro API key header if available
    headers = {}
    if COINGECKO_PRO_API_KEY:
        headers["x-cg-pro-api-key"] = COINGECKO_PRO_API_KEY

    try:
        response = httpx.get(api_url, params=params, headers=headers, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Fetched {len(data)} coins from CoinGecko{' Pro' if COINGECKO_PRO_API_KEY else ''}")
        
        stream_progress_event(
            writer,
            stage="fetch_market_data",
            status="completed",
            message=f"{len(data)} ativos carregados",
            order=1,
            total_steps=4,
            details={"count": len(data)},
        )
        
        # Resumo para o LLM
        top_coins = [c['symbol'].upper() for c in data[:5]]
        summary = f"✅ {len(data)} moedas carregadas no estado. Top 5: {', '.join(top_coins)}..."
        
        return Command(
            update={
                "market_data": data,
                "messages": [ToolMessage(
                    content=summary,
                    tool_call_id=tool_call_id
                )]
            }
        )
        
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            print(f"⚠️ Rate limit exceeded. Consider using frontend cache/proxy.")
        print(f"❌ Error fetching market data: {e}")
        stream_progress_event(
            writer,
            stage="fetch_market_data",
            status="error",
            message=f"Erro ao buscar dados de mercado: {e}",
            order=1,
            total_steps=4,
        )
        return Command(
            update={
                "messages": [ToolMessage(
                    content=f"Erro ao buscar dados: {str(e)}",
                    tool_call_id=tool_call_id
                )]
            }
        )
    except Exception as e:
        print(f"❌ Error fetching market data: {e}")
        stream_progress_event(
            writer,
            stage="fetch_market_data",
            status="error",
            message=f"Erro ao buscar dados de mercado: {e}",
            order=1,
            total_steps=4,
        )
        return Command(
            update={
                "messages": [ToolMessage(
                    content=f"Erro inesperado: {str(e)}",
                    tool_call_id=tool_call_id
                )]
            }
        )

@tool
def get_coin_details(
    coin_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """
    Busca detalhes de uma criptomoeda específica
    
    **NOTE**: This tool uses CoinGecko API directly. Use sparingly to avoid rate limits.

    Args:
        coin_id: ID da moeda (ex: bitcoin, ethereum)

    Returns:
        Command com os detalhes
    """
    api_url = f"{COINGECKO_BASE_URL}/coins/{coin_id}"
    params = {
        "localization": False,
        "tickers": False,
        "community_data": False,
        "developer_data": False
    }

    # Add Pro API key header if available
    headers = {}
    if COINGECKO_PRO_API_KEY:
        headers["x-cg-pro-api-key"] = COINGECKO_PRO_API_KEY

    try:
        response = httpx.get(api_url, params=params, headers=headers, timeout=10.0)
        response.raise_for_status()
        data = response.json()
        
        return Command(
            update={
                "messages": [ToolMessage(
                    content=f"✅ Detalhes de {coin_id} obtidos. Preço: ${data.get('market_data', {}).get('current_price', {}).get('usd', 'N/A')}",
                    tool_call_id=tool_call_id
                )]
            }
        )
    except Exception as e:
        print(f"❌ Error fetching coin details: {e}")
        return Command(
            update={
                "messages": [ToolMessage(
                    content=f"Erro ao buscar detalhes: {str(e)}",
                    tool_call_id=tool_call_id
                )]
            }
        )

def get_market_tools():
    """Retorna lista de ferramentas de mercado"""
    return [fetch_market_data, get_coin_details]
