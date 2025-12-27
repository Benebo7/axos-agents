"""Ferramentas para geração e comparação de relatórios de análise"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Annotated

from langchain.tools import tool, InjectedToolCallId
from langgraph.prebuilt.tool_node import InjectedState
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langchain_openai import ChatOpenAI
import os

from tools.stream_utils import (
    get_stream_writer_safe,
    stream_custom_event,
    stream_progress_event,
)


def format_opportunities_summary(opportunities: List[Dict]) -> str:
    """Formata oportunidades para o prompt do LLM."""
    if not opportunities:
        return "Nenhuma oportunidade identificada."

    summary_lines = []
    for opp in opportunities[:5]:  # Top 5
        coin = opp.get("coin_symbol", "N/A").upper()
        action = opp.get("action", "N/A")
        confidence = opp.get("confidence", 0)
        expected_return = opp.get("expected_return", 0)

        summary_lines.append(
            f"- {coin}: {action} (Confiança: {confidence:.0f}%, "
            f"Retorno esperado: {expected_return*100:+.1f}%)"
        )

    return "\n".join(summary_lines)


def format_tasks_summary(tasks: List[Dict]) -> str:
    """Formata tasks para o prompt do LLM."""
    if not tasks:
        return "Nenhuma task recomendada."

    summary_lines = []
    for task in tasks[:5]:  # Top 5
        title = task.get("title", "N/A")
        priority = task.get("priority", "medium")

        summary_lines.append(f"- [{priority.upper()}] {title}")

    return "\n".join(summary_lines)


def format_allocation_summary(allocation: Dict) -> str:
    """Formata alocação para o prompt do LLM."""
    if not allocation or "allocations" not in allocation:
        return "Nenhuma alocação definida."

    allocations = allocation.get("allocations", [])
    if not allocations:
        return "Nenhuma alocação definida."

    summary_lines = []
    for alloc in allocations:
        coin = alloc.get("coin_symbol", "N/A").upper()
        percentage = alloc.get("percentage", 0)
        amount = alloc.get("amount_usd", 0)

        summary_lines.append(f"- {coin}: {percentage:.1f}% (${amount:,.2f})")

    return "\n".join(summary_lines)


def get_current_price(coin_symbol: str) -> float:
    """
    Busca preço atual de uma moeda.
    Em produção, isso deveria chamar a API CoinGecko.
    Por ora, retorna um placeholder.
    """
    # TODO: Integrar com CoinGecko API
    # Por enquanto, retorna um valor simulado
    import random
    base_prices = {
        "btc": 95000,
        "eth": 3200,
        "sol": 110,
        "xrp": 0.60,
        "ada": 0.45,
        "xrp": 0.60,
    }

    base_price = base_prices.get(coin_symbol.lower(), 100)
    # Adiciona variação aleatória de ±5%
    variation = random.uniform(-0.05, 0.05)
    return base_price * (1 + variation)


def analyze_market_changes(baseline: Dict, current_opportunities: List[Dict]) -> Dict:
    """
    Analisa mudanças de mercado entre baseline e análise atual.
    """
    # Análise simplificada
    baseline_opps = baseline.get("opportunities", [])

    # Calcular mudança de sentimento (simplificado)
    baseline_avg_confidence = (
        sum(o.get("confidence", 0) for o in baseline_opps) / len(baseline_opps)
        if baseline_opps else 50
    )
    current_avg_confidence = (
        sum(o.get("confidence", 0) for o in current_opportunities) / len(current_opportunities)
        if current_opportunities else 50
    )

    confidence_change = current_avg_confidence - baseline_avg_confidence

    if confidence_change > 10:
        sentiment_shift = "improved"
    elif confidence_change < -10:
        sentiment_shift = "worsened"
    else:
        sentiment_shift = "stable"

    return {
        "sentiment_shift": sentiment_shift,
        "confidence_change": confidence_change,
        "volatility_change": 0,  # Placeholder
        "volume_change": 0,  # Placeholder
    }


@tool
def generate_executive_report(
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
    state: Annotated[dict, InjectedState] = None,
) -> Command:
    """
    Gera relatório executivo narrativo da análise completa.

    Sintetiza opportunities, tasks e allocation em:
    - Executive Summary (3-5 frases)
    - Market Context (sentimento, tendências)
    - Key Insights (3-5 bullet points)
    - Warnings/Risks (alertas críticos)
    """
    print("📊 generate_executive_report iniciado")

    # Acessar estado atual
    if not state:
        state = {}
        
    opportunities = state.get("opportunities", [])
    tasks = state.get("tasks", [])
    allocation = state.get("allocation", {})
    mode = state.get("mode", "market_analysis")

    print(f"   Oportunidades: {len(opportunities)}")
    print(f"   Tasks: {len(tasks)}")
    print(f"   Modo: {mode}")

    # Construir prompt para LLM
    report_prompt = f"""Baseado na análise de mercado completa:

**Oportunidades identificadas**: {len(opportunities)}
{format_opportunities_summary(opportunities)}

**Tasks recomendadas**: {len(tasks)}
{format_tasks_summary(tasks)}

**Alocação sugerida**:
{format_allocation_summary(allocation)}

Gere um relatório executivo profissional em português com:

1. **Executive Summary** (máx 5 frases): Resumo geral da análise
2. **Market Context** (2-3 frases): Sentimento atual do mercado
3. **Key Insights** (3-5 bullet points): Principais descobertas acionáveis
4. **Warnings** (2-4 bullet points): Riscos e alertas importantes

Formato JSON ESTRITO (sem markdown, apenas JSON puro):
{{
  "executive_summary": "...",
  "market_context": "...",
  "key_insights": ["...", "...", "..."],
  "warnings": ["...", "..."]
}}
"""

    try:
        # Gerar relatório via LLM
        model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
        )

        response = model.invoke(report_prompt)

        # Extrair JSON da resposta
        content = response.content

        # Remover markdown code blocks se existirem
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        report_data = json.loads(content)

        print(f"✅ Relatório gerado com sucesso")
        print(f"   Summary: {report_data.get('executive_summary', '')[:100]}...")

    except Exception as e:
        print(f"⚠️ Erro ao gerar relatório: {e}")
        # Fallback para relatório básico
        report_data = {
            "executive_summary": f"Análise identificou {len(opportunities)} oportunidades de investimento com base no perfil de risco selecionado. Recomendações incluem diversificação entre {len(allocation.get('allocations', []))} ativos principais.",
            "market_context": "Mercado apresenta condições normais de operação com volatilidade moderada.",
            "key_insights": [
                f"Identificadas {len(opportunities)} oportunidades de investimento",
                f"Sugeridas {len(tasks)} ações prioritárias",
                "Diversificação recomendada entre principais criptomoedas"
            ],
            "warnings": [
                "Mercado de criptomoedas apresenta alta volatilidade",
                "Sempre faça sua própria pesquisa antes de investir"
            ]
        }

    # Streaming de progresso
    writer = get_stream_writer_safe()
    stream_progress_event(
        writer,
        stage="generate_report",
        status="completed",
        message="Relatório executivo gerado",
        order=5,
        total_steps=5
    )

    # Atualizar estado com relatório - incluindo ToolMessage obrigatório
    return Command(
        update={
            "executive_report": report_data,
            "analysis_complete": True,
            "messages": [
                ToolMessage(
                    content=f"Relatório executivo gerado com sucesso. Summary: {report_data.get('executive_summary', '')[:100]}...",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool
def load_baseline_report(
    baseline_report_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> Command:
    """
    Carrega relatório baseline do localStorage (via frontend).

    Esta tool retorna um placeholder e espera que o frontend
    injete os dados do baseline no próximo invoke.

    Args:
        baseline_report_id: ID do relatório baseline a ser carregado
    """
    print(f"📂 load_baseline_report iniciado: {baseline_report_id}")

    writer = get_stream_writer_safe()
    stream_custom_event(
        writer,
        "request_baseline",
        {"baseline_report_id": baseline_report_id}
    )

    stream_progress_event(
        writer,
        stage="load_baseline",
        status="completed",
        message="Solicitação de baseline enviada ao frontend",
        order=1,
        total_steps=5
    )

    print(f"✅ Evento request_baseline enviado para frontend")

    # Frontend deve responder com baseline_data no próximo invoke
    return Command(
        update={
            "baseline_requested": True,
            "baseline_report_id": baseline_report_id,
            "messages": [
                ToolMessage(
                    content=f"Baseline report {baseline_report_id} solicitado. Aguardando dados do frontend.",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )


@tool
def compare_with_baseline(
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
    state: Annotated[dict, InjectedState] = None,
) -> Command:
    """
    Compara análise atual com baseline report.

    Gera:
    - Performance de cada recomendação (PnL)
    - Mudanças de mercado (sentimento, volatilidade)
    - Drift de alocação
    - Alertas (danger/warning/opportunity)
    """
    print("🔄 compare_with_baseline iniciado")

    # Acessar estado via InjectedState
    if not state:
        state = {}
        
    baseline = state.get("baseline_data", {})
    current_opportunities = state.get("opportunities", [])

    if not baseline:
        print("⚠️ Baseline data não encontrado no estado")
        return Command(
            update={
                "comparison": {
                    "recommendations_performance": [],
                    "alerts": [{
                        "type": "warning",
                        "message": "Baseline data não disponível",
                        "affected_coins": [],
                        "suggested_action": "Verifique se o relatório baseline existe"
                    }],
                    "market_changes": {}
                },
                "messages": [
                    ToolMessage(
                        content="Comparação não realizada: baseline data não disponível.",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    print(f"   Baseline opportunities: {len(baseline.get('opportunities', []))}")
    print(f"   Current opportunities: {len(current_opportunities)}")

    # Calcular performance de recomendações
    recommendations_performance = []
    baseline_opportunities = baseline.get("opportunities", [])

    for baseline_opp in baseline_opportunities:
        coin_symbol = baseline_opp.get("coin_symbol", "")
        entry_price = baseline_opp.get("entry_price", 0)

        if not coin_symbol or not entry_price:
            continue

        current_price = get_current_price(coin_symbol)
        pnl_percent = ((current_price - entry_price) / entry_price) * 100

        target_price = baseline_opp.get("target_price")
        stop_loss = baseline_opp.get("stop_loss")

        recommendations_performance.append({
            "coin": coin_symbol.upper(),
            "action": baseline_opp.get("action", "HOLD"),
            "entry_price": entry_price,
            "current_price": current_price,
            "pnl_percent": pnl_percent,
            "target_reached": current_price >= target_price if target_price else False,
            "stop_loss_hit": current_price <= stop_loss if stop_loss else False
        })

    print(f"   Performance calculada para {len(recommendations_performance)} recomendações")

    # Detectar alertas
    alerts = []
    for perf in recommendations_performance:
        if perf["stop_loss_hit"]:
            alerts.append({
                "type": "danger",
                "message": f"{perf['coin']} atingiu stop loss",
                "affected_coins": [perf["coin"]],
                "suggested_action": "Considere sair da posição imediatamente"
            })
        elif perf["target_reached"]:
            alerts.append({
                "type": "opportunity",
                "message": f"{perf['coin']} atingiu target price",
                "affected_coins": [perf["coin"]],
                "suggested_action": "Considere realizar lucros parciais"
            })
        elif abs(perf["pnl_percent"]) > 10:
            alert_type = "opportunity" if perf["pnl_percent"] > 0 else "warning"
            alerts.append({
                "type": alert_type,
                "message": f"{perf['coin']}: {perf['pnl_percent']:+.1f}%",
                "affected_coins": [perf["coin"]],
                "suggested_action": "Revisar posição e ajustar stop loss se necessário"
            })

    print(f"   Alertas gerados: {len(alerts)}")

    # Analisar mudanças de mercado
    market_changes = analyze_market_changes(baseline, current_opportunities)

    comparison = {
        "recommendations_performance": recommendations_performance,
        "alerts": alerts,
        "market_changes": market_changes
    }

    writer = get_stream_writer_safe()
    stream_progress_event(
        writer,
        stage="compare_baseline",
        status="completed",
        message="Comparação com baseline concluída",
        order=3,
        total_steps=5
    )

    print(f"✅ Comparação concluída")

    return Command(
        update={
            "comparison": comparison,
            "messages": [
                ToolMessage(
                    content=f"Comparação concluída: {len(recommendations_performance)} recomendações analisadas, {len(alerts)} alertas gerados.",
                    tool_call_id=tool_call_id
                )
            ]
        }
    )
