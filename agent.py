"""
Deep Agent para Análise de Criptomoedas
Usa LangGraph com streaming para UI em tempo real
"""
from typing import List, Annotated, Sequence
from typing_extensions import NotRequired
from langchain.agents import create_agent, AgentState
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from tools.market_tools import get_market_tools
from tools.analysis_tools import get_analysis_tools
from tools.report_tools import generate_executive_report
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente do .env (incluindo LangSmith tracing)
load_dotenv()

# Estado do agente - DEVE estender AgentState
class AnalysisState(AgentState):
    """
    Estado do agente de análise com campos customizados.
    Suporta 2 modos: market_analysis, single_coin_analysis
    """

    # ========================================
    # COMUM (todos os modos)
    # ========================================
    messages: Annotated[Sequence[BaseMessage], add_messages]
    mode: NotRequired[str]  # 'market_analysis' | 'single_coin_analysis'

    # ========================================
    # MODO: single_coin_analysis
    # ========================================
    coin_id: NotRequired[str]   # ex: 'bitcoin', 'ethereum'
    period: NotRequired[str]    # ex: '7d', '30d', '90d'

    # ========================================
    # MODO: market_analysis
    # ========================================
    user_profile: NotRequired[dict]
    # {
    #   experience: str,           # nível de experiência
    #   riskTolerance: str,        # perfil de risco
    #   objectives: str,           # objetivo geral
    #   volatilityReaction: str,   # reação a volatilidade
    #   financialHealth: str,      # saúde financeira
    #   raw: dict                  # questionário completo
    # }

    analysis_context: NotRequired[dict]
    # {
    #   timeHorizon: str,                  # horizonte de tempo
    #   objective: str,                    # objetivo específico
    #   capital: float,                    # capital a investir
    #   capitalRepresentationPercent: int  # % do patrimônio
    # }

    portfolio: NotRequired[dict]
    # {
    #   assets: [{ coin_id, amount, price_usd }]
    # }

    # ========================================
    # SAÍDA: single_coin_analysis
    # ========================================
    opportunity: NotRequired[dict]
    # {
    #   opportunity_id: str,
    #   coin_id: str,
    #   coin_symbol: str,
    #   coin_name: str,
    #   confidence: float,
    #   tag: str,
    #   reason: str,
    #   analysis: str,
    #   entry_price: float,
    #   target_price: float,
    #   stop_loss: float,
    #   risk_level: str,
    #   timeframe: str
    # }

    # ========================================
    # SAÍDA: market_analysis
    # ========================================
    market_data: NotRequired[List[dict]]
    opportunities: NotRequired[List[dict]]
    tasks: NotRequired[List[dict]]
    allocation: NotRequired[dict]
    executive_report: NotRequired[dict]
    # {
    #   executive_summary: str,
    #   market_context: str,
    #   key_insights: List[str],
    #   warnings: List[str]
    # }

    # ========================================
    # COMUM (flag de conclusão)
    # ========================================
    analysis_complete: NotRequired[bool]

# Configuração do modelo usando OpenRouter
model = ChatOpenAI(
    model="anthropic/claude-4.5-sonnet",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", ""),
    max_tokens=8000,  # Limitar tokens para evitar erro de créditos
    temperature=0.7
)

# System prompt do agente
SYSTEM_PROMPT = """Você é um agente especializado em análise de criptomoedas.

**IMPORTANTE SOBRE ESTADO:**
- Todas as tools SALVAM seus resultados no ESTADO do grafo.
- Você NÃO precisa passar dados grandes (JSONs) entre tools.
- As tools acessam automaticamente `market_data`, `opportunities`, etc. do estado.
- Apenas chame as tools na ordem correta, sem passar os dados gerados anteriormente como argumentos.

**VERIFIQUE O MODO DE OPERAÇÃO NO ESTADO:**
`mode` pode ser "market_analysis" (Geral) ou "single_coin_analysis" (Ativo Único).

---

### 🅰️ MODO GERAL (mode="market_analysis")
Siga este fluxo se o usuário quer uma análise geral do mercado:

1️⃣ **Buscar dados do mercado**
   ```python
   fetch_market_data(period="7d", limit=30)
   # ✅ market_data agora está no estado
   ```

2️⃣ **Analisar oportunidades**
   ```python
   analyze_opportunities(risk_profile=state["risk_profile"])
   # ✅ Pega market_data do estado automaticamente
   # ✅ opportunities agora está no estado
   ```

3️⃣ **Gerar tarefas & Alocação** (Em paralelo ou sequencial)
   ```python
   generate_tasks(capital=state["capital"]) # ✅ Pega opportunities e capital do estado
   create_allocation(capital=state["capital"], risk_profile=state["risk_profile"]) # ✅ Pega opportunities do estado
   ```

4️⃣ **FINALIZAR: Gerar relatório executivo** (OBRIGATÓRIO)
   Você DEVE chamar esta ferramenta para completar a análise.
   Não passe argumentos.
   ```python
   generate_executive_report()
   # ✅ Pega tudo do estado e gera o relatório final
   ```

---

### 🅱️ MODO ATIVO ÚNICO (mode="single_coin_analysis")
Siga este fluxo se `coin_id` estiver preenchido (ex: "bitcoin"):

1️⃣ **Buscar detalhes do ativo**
   ```python
   get_coin_details(coin_id=state["coin_id"])
   ```

2️⃣ **Analisar o ativo** (FINALIZA AUTOMATICAMENTE)
   ```python
   analyze_opportunities()
   # ✅ Retorna `opportunity` (objeto único) e marca analysis_complete=True
   # ✅ NÃO chame generate_tasks, create_allocation ou generate_executive_report
   ```

---

**REGRAS ABSOLUTAS:**
✅ NÃO invente dados. Use as tools.
✅ NÃO passe JSONs gigantes como argumento. Confie no estado.
✅ Se uma tool falhar, tente recuperar ou notificar o erro.
✅ MODO GERAL: A última ação DEVE SER `generate_executive_report`.
✅ MODO SINGLE: A última ação é `analyze_opportunities` (finaliza automaticamente).
"""

# Ferramentas disponíveis
tools = [
    *get_market_tools(),
    *get_analysis_tools(),
    generate_executive_report,
]

# Criar agent com state_schema para persistir estado customizado
print("🔧 Criando crypto analysis agent...")

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    state_schema=AnalysisState,
    name="crypto_analysis_agent"
)

# Compilar o grafo
graph = agent

print("✅ Agent compilado e pronto!")
print(f"📋 Tools disponíveis: {[t.name for t in tools]}")
print(f"🔒 Limite de concorrência ativo via execution_limiter (env: MAX_CONCURRENT_EXECUTIONS)")
