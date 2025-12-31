"""
Agent para Follow-up/Report de Análises
Compara análise original com estado atual do mercado
"""
from typing import List, Annotated, Sequence
from typing_extensions import NotRequired
from langchain.agents import create_agent, AgentState
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from tools.market_tools import get_market_tools
from tools.report_tools import (
    generate_executive_report,
    load_baseline_report,
    compare_with_baseline
)
import os
from dotenv import load_dotenv

load_dotenv()


# Estado do agente de report
class ReportState(AgentState):
    """
    Estado do agente de follow-up/report.
    Recebe análise original e gera comparação com mercado atual.
    """

    # ========================================
    # COMUM
    # ========================================
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # ========================================
    # ENTRADA (vem do BFF)
    # ========================================
    original_analysis: NotRequired[dict]
    # {
    #   id: str,                  # ID da análise
    #   request_payload: dict,    # questionário + form original
    #   result_payload: dict,     # opportunities, allocation gerados
    #   created_at: str
    # }

    previous_reports: NotRequired[List[dict]]
    # [
    #   { id, created_at, summary, performance_snapshot },
    #   ...
    # ]

    # ========================================
    # PROCESSAMENTO (usado internamente)
    # ========================================
    baseline_report_id: NotRequired[str]
    baseline_data: NotRequired[dict]
    baseline_requested: NotRequired[bool]
    current_market_data: NotRequired[List[dict]]
    comparison: NotRequired[dict]

    # ========================================
    # SAÍDA
    # ========================================
    report: NotRequired[dict]
    # {
    #   summary: str,
    #   performance: {
    #     recommendations_performance: [...],
    #     overall_pnl: float
    #   },
    #   market_changes: {...},
    #   alerts: [...],
    #   next_actions: [...]
    # }
    report_complete: NotRequired[bool]


# Configuração do modelo
model = ChatOpenAI(
    model="anthropic/claude-4.5-sonnet",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", ""),
    max_tokens=8000,
    temperature=0.7
)

# System prompt
REPORT_SYSTEM_PROMPT = """Você é um agente especializado em gerar relatórios de acompanhamento de análises de criptomoedas.

**SEU OBJETIVO:**
Comparar uma análise anterior (original_analysis) com o estado atual do mercado e gerar um relatório de performance.

**DADOS DISPONÍVEIS NO ESTADO:**
- `original_analysis`: A análise original com opportunities e allocation recomendados
- `previous_reports`: Histórico de reports anteriores (se houver)

**FLUXO:**

1️⃣ **Carregar baseline**
   ```python
   load_baseline_report(baseline_report_id=<id da análise original>)
   ```

2️⃣ **Buscar dados atuais do mercado**
   ```python
   fetch_market_data()
   ```

3️⃣ **Comparar com baseline**
   ```python
   compare_with_baseline()
   # Gera: recommendations_performance, alerts, market_changes
   ```

4️⃣ **Gerar relatório executivo**
   ```python
   generate_executive_report()
   ```

**REGRAS:**
- Use os dados da análise original para comparar
- Calcule PnL de cada recomendação
- Identifique alertas (stop loss, target reached)
- Sugira próximas ações
"""

# Ferramentas
tools = [
    *get_market_tools(),
    generate_executive_report,
    load_baseline_report,
    compare_with_baseline,
]

# Criar agent
print("🔧 Criando report agent...")

report_agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=REPORT_SYSTEM_PROMPT,
    state_schema=ReportState,
    name="report_agent"
)

report_graph = report_agent

print("✅ Report Agent compilado e pronto!")
