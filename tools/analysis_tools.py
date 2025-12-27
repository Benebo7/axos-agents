"""Ferramentas para análise técnica e geração de insights"""
from langchain.tools import tool, InjectedToolCallId
from langgraph.prebuilt.tool_node import InjectedState
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from typing import List, Dict, Union, Optional, Annotated
import time
import json

from tools.stream_utils import (
    get_stream_writer_safe,
    stream_custom_event,
    stream_progress_event,
)

@tool
def analyze_opportunities(
    market_data_json: Optional[Union[str, List[Dict], Dict]] = None,
    risk_profile: str = "moderate",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
    state: Annotated[dict, InjectedState] = None,
) -> Command:
    """
    Analisa dados de mercado e identifica oportunidades de investimento.

    IMPORTANTE: Esta ferramenta AUTOMATICAMENTE atualiza o estado do grafo
    com as oportunidades identificadas via Command.

    Args:
        market_data_json: (Opcional/Deprecado) Dados do mercado. 
                         PREFERENCIALMENTE usa state['market_data'].
        risk_profile: Perfil de risco ("conservative", "moderate", "aggressive")

    Returns:
        Command que atualiza o estado com as oportunidades
    """
    print("🔍 analyze_opportunities iniciado")
    print(f"   tool_call_id: {tool_call_id}")
    
    # ✅ Obter stream writer para streaming em tempo real
    writer = get_stream_writer_safe()
    stream_progress_event(
        writer,
        stage="analyze_opportunities",
        status="running",
        message="Identificando oportunidades em tempo real",
        order=2,
        total_steps=4,
        details={"risk_profile": risk_profile},
    )

    market_data = None
    
    # 1. Tentar ler do estado (PRIORIDADE)
    if state and "market_data" in state and state["market_data"]:
        market_data = state["market_data"]
        print(f"   ✅ Usando market_data do estado: {len(market_data)} itens")
    
    # 2. Se não tem no estado, verificar argumento (Retrocompatibilidade)
    elif market_data_json:
        if isinstance(market_data_json, str):
            try:
                market_data = json.loads(market_data_json)
            except:
                pass
        elif isinstance(market_data_json, (list, dict)):
            market_data = market_data_json
        
        # Normalizar para lista
        if isinstance(market_data, dict):
            market_data = [market_data]
            
    # 3. Se nada funcionar, tentar buscar
    if not market_data:
        print("⚠️  market_data não encontrado. Buscando automaticamente...")
        # Nota: Não podemos chamar tools diretamente aqui se elas retornam Command.
        error_msg = "Dados de mercado não encontrados no estado. Por favor, chame fetch_market_data primeiro."
        return Command(
            update={
                "messages": [ToolMessage(
                    content=error_msg,
                    tool_call_id=tool_call_id
                )]
            }
        )

    # ========================================
    # ANÁLISE DAS OPORTUNIDADES
    # ========================================

    opportunities = []

    # Critérios baseados no perfil de risco
    risk_thresholds = {
        "conservative": {"min_market_cap": 10_000_000_000, "max_volatility": 5},
        "moderate": {"min_market_cap": 1_000_000_000, "max_volatility": 10},
        "aggressive": {"min_market_cap": 100_000_000, "max_volatility": 20}
    }

    threshold = risk_thresholds.get(risk_profile, risk_thresholds["moderate"])

    print(f"🎯 Analisando {len(market_data)} moedas com perfil {risk_profile}")

    for coin in market_data:
        price_change = coin.get("price_change_percentage_24h", 0)
        # Handle None price_change
        if price_change is None:
            price_change = 0
            
        market_cap = coin.get("market_cap", 0) or 0
        volume = coin.get("total_volume", 0) or 0
        current_price = coin.get("current_price", 0) or 0

        if market_cap >= threshold["min_market_cap"] and abs(price_change) <= threshold["max_volatility"]:
            confidence = min(95, 70 + abs(price_change) * 2)
            tag = _determine_tag(price_change)
            risk_level = _determine_risk_level(market_cap, abs(price_change))

            opportunity = {
                "opportunity_id": f"{coin['id']}_{int(time.time() * 1000)}",
                "coin_id": coin["id"],
                "coin_symbol": coin["symbol"].upper(),
                "coin_name": coin["name"],
                "confidence": float(confidence),
                "tag": tag,
                "reason": f"{'Reversão detectada' if price_change < -5 else 'Momentum positivo'} com volume {volume:,.0f}",
                "analysis": f"Análise detalhada de {coin['name']}: Market cap ${market_cap:,.0f}, mudança 24h {price_change:.2f}%, volume ${volume:,.0f}. Perfil {risk_profile} identifica esta como oportunidade {tag.lower()}.",
                "entry_price": float(current_price),
                "target_price": float(current_price * 1.15),
                "stop_loss": float(current_price * 0.95),
                "risk_level": risk_level,
                "timeframe": "2-4 semanas"
            }

            opportunities.append(opportunity)
            # ✅ Enviar oportunidade imediatamente via streaming custom
            stream_custom_event(writer, "opportunity", opportunity)

    result_opportunities = opportunities[:10]  # Top 10
    print(f"🎉 Análise completa: {len(result_opportunities)} oportunidades identificadas")

    # ========================================
    # ATUALIZAR ESTADO DO GRAFO VIA COMMAND
    # ========================================

    stream_progress_event(
        writer,
        stage="analyze_opportunities",
        status="completed",
        message=f"{len(result_opportunities)} oportunidades identificadas",
        order=2,
        total_steps=4,
        details={"count": len(result_opportunities)},
    )

    return Command(
        update={
            # ✅ CRÍTICO: Atualizar o estado com as oportunidades
            "opportunities": result_opportunities,
            # ✅ OBRIGATÓRIO: Incluir ToolMessage na message history
            "messages": [ToolMessage(
                content=f"✅ Identificadas {len(result_opportunities)} oportunidades com perfil {risk_profile}. (Salvas no estado)",
                tool_call_id=tool_call_id
            )]
        }
    )


@tool
def generate_tasks(
    opportunities_json: Optional[Union[str, List[Dict]]] = None,
    capital: float = 10000,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
    state: Annotated[dict, InjectedState] = None,
) -> Command:
    """
    Gera tarefas personalizadas baseadas nas oportunidades
    
    Args:
        opportunities_json: (Opcional) Usa state['opportunities'] preferencialmente.
        capital: Capital disponível do usuário

    Returns:
        Command que atualiza o estado com as tarefas
    """
    print("📋 generate_tasks iniciado")
    
    # ✅ Obter stream writer para streaming em tempo real
    writer = get_stream_writer_safe()
    stream_progress_event(
        writer,
        stage="generate_tasks",
        status="running",
        message="Gerando tarefas operacionais",
        order=3,
        total_steps=4,
    )

    # 1. Tentar ler do estado (PRIORIDADE)
    opportunities = []
    if state and "opportunities" in state and state["opportunities"]:
        opportunities = state["opportunities"]
        print(f"   ✅ Usando opportunities do estado: {len(opportunities)} itens")
    
    # 2. Fallback para argumento
    elif opportunities_json:
        if isinstance(opportunities_json, str):
            try:
                opportunities = json.loads(opportunities_json)
            except:
                pass
        elif isinstance(opportunities_json, list):
            opportunities = opportunities_json

    tasks = []

    if opportunities:
        tasks.append({
            "id": f"review_opp_{int(time.time())}",
            "title": "Revisar Oportunidades Identificadas",
            "description": f"Analisar {len(opportunities)} oportunidades detectadas pela IA",
            "priority": "alta",
            "type": "review",
            "due_date": "2024-12-20",
            "progress": 0,
            "completed": False
        })

    tasks.append({
        "id": f"diversify_{int(time.time())}",
        "title": "Diversificar Portfólio",
        "description": "Considerar diversificação entre as top oportunidades identificadas",
        "priority": "média",
        "type": "action",
        "due_date": "2024-12-25",
        "progress": 0,
        "completed": False
    })

    # ✅ Stream cada task imediatamente
    for task in tasks:
        stream_custom_event(writer, "task", task)

    stream_progress_event(
        writer,
        stage="generate_tasks",
        status="completed",
        message=f"{len(tasks)} tarefas geradas",
        order=3,
        total_steps=4,
        details={"count": len(tasks)},
    )

    return Command(
        update={
            "tasks": tasks,
            "messages": [ToolMessage(
                content=f"✅ Geradas {len(tasks)} tarefas. (Salvas no estado)",
                tool_call_id=tool_call_id
            )]
        }
    )


@tool
def create_allocation(
    capital: float,
    opportunities_json: Optional[Union[str, List[Dict]]] = None,
    risk_profile: str = "moderate",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
    state: Annotated[dict, InjectedState] = None,
) -> Command:
    """
    Cria alocação otimizada de portfólio
    
    Args:
        capital: Capital disponível
        opportunities_json: (Opcional) Usa state['opportunities'] preferencialmente.
        risk_profile: Perfil de risco

    Returns:
        Command que atualiza o estado com a alocação
    """
    print("💰 create_allocation iniciado")
    
    # ✅ Obter stream writer para streaming em tempo real
    writer = get_stream_writer_safe()
    stream_progress_event(
        writer,
        stage="create_allocation",
        status="running",
        message="Calculando alocação recomendada",
        order=4,
        total_steps=4,
        details={"capital": capital, "risk_profile": risk_profile},
    )

    # 1. Tentar ler do estado (PRIORIDADE)
    opportunities = []
    if state and "opportunities" in state and state["opportunities"]:
        opportunities = state["opportunities"]
        print(f"   ✅ Usando opportunities do estado: {len(opportunities)} itens")
    elif opportunities_json:
        if isinstance(opportunities_json, str):
             try:
                opportunities = json.loads(opportunities_json)
             except:
                pass
        elif isinstance(opportunities_json, list):
            opportunities = opportunities_json

    allocation_strategy = {
        "conservative": {"crypto": 0.4, "stable": 0.6},
        "moderate": {"crypto": 0.6, "stable": 0.4},
        "aggressive": {"crypto": 0.8, "stable": 0.2}
    }

    strategy = allocation_strategy.get(risk_profile, allocation_strategy["moderate"])
    crypto_capital = capital * strategy["crypto"]

    allocations = []
    if opportunities:
        per_coin = crypto_capital / min(len(opportunities), 5)

        for opp in opportunities[:5]:
            allocations.append({
                "coin_id": opp["coin_id"],
                "coin_symbol": opp["coin_symbol"],
                "percentage": (per_coin / capital) * 100,
                "reason": f"Alta confiança ({opp['confidence']:.0f}%) identificada pela IA",
                "risk": opp["risk_level"].lower(),
                "expected_return": "15-25%",
                "timeframe": opp["timeframe"]
            })

    allocation = {
        "allocation": allocations,
        "risk_metrics": {
            "portfolio_volatility": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.18
        },
        "rebalancing_suggestions": [],
        "total_expected_return": "18-30%",
        "diversification_score": 0.75
    }

    # ✅ Stream allocation imediatamente
    stream_custom_event(writer, "allocation", allocation)

    stream_progress_event(
        writer,
        stage="create_allocation",
        status="completed",
        message="Alocação recomendada calculada",
        order=4,
        total_steps=4,
        details={"assets": len(allocations)},
    )
    
    # Stream complete event
    stream_custom_event(
        writer,
        "complete",
        {
            "status": "success",
            "timestamp": time.time(),
        },
    )

    return Command(
        update={
            "allocation": allocation,
            # analysis_complete removido daqui - deve ser setado apenas no report final
            "messages": [ToolMessage(
                content=f"✅ Alocação otimizada criada com {len(allocations)} moedas. (Salva no estado)",
                tool_call_id=tool_call_id
            )]
        }
    )


def _determine_tag(price_change: float) -> str:
    """Determina a tag baseada na mudança de preço"""
    if price_change > 10:
        return "Rápida"
    elif price_change > 5:
        return "Momentum"
    elif price_change < -5:
        return "Reversão"
    elif abs(price_change) < 2:
        return "Evento"
    else:
        return "Gem"


def _determine_risk_level(market_cap: float, volatility: float) -> str:
    """Determina nível de risco baseado em market cap e volatilidade"""
    if market_cap > 10_000_000_000 and volatility < 5:
        return "Baixo"
    elif market_cap > 1_000_000_000 and volatility < 10:
        return "Médio"
    else:
        return "Alto"


def get_analysis_tools():
    """Retorna lista de ferramentas de análise"""
    return [
        analyze_opportunities,
        generate_tasks,
        create_allocation
    ]
