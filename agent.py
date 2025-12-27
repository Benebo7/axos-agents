"""
Deep Agent para Análise de Criptomoedas
Usa LangGraph com streaming para UI em tempo real
"""
from typing import List, Dict, Any, Annotated, Sequence
from typing_extensions import NotRequired
from langchain.agents import create_agent, AgentState
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END
from langgraph.graph.message import add_messages
from langgraph.types import Command

from tools.market_tools import get_market_tools
from tools.analysis_tools import get_analysis_tools
from tools.report_tools import (
    generate_executive_report,
    load_baseline_report,
    compare_with_baseline
)
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente do .env (incluindo LangSmith tracing)
load_dotenv()

# Estado do agente - DEVE estender AgentState
class AnalysisState(AgentState):
    """Estado do agente de análise com campos customizados"""
    # ✅ Garante que o histórico de mensagens seja preservado (append)
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
    # Campos customizados (NotRequired para não serem obrigatórios no invoke)
    user_id: NotRequired[str]
    period: NotRequired[str]
    risk_profile: NotRequired[str]
    capital: NotRequired[float]
    mode: NotRequired[str]
    coin_id: NotRequired[str]
    
    # ✅ Dados principais
    market_data: NotRequired[List[dict]]
    opportunities: NotRequired[List[dict]]
    tasks: NotRequired[List[dict]]
    allocation: NotRequired[dict]
    analysis_complete: NotRequired[bool]

    # Campos para reports
    executive_report: NotRequired[dict]  # { executive_summary, market_context, key_insights, warnings }
    baseline_requested: NotRequired[bool]
    baseline_report_id: NotRequired[str]
    baseline_data: NotRequired[dict]  # Injetado pelo frontend
    comparison: NotRequired[dict]  # Resultado de compare_with_baseline

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
`mode` pode ser "market_analysis" (Geral), "single_coin_analysis" (Ativo Único) ou "follow_up_analysis" (Follow-Up).

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

2️⃣ **Analisar o ativo**
   ```python
   analyze_opportunities(risk_profile=state["risk_profile"])
   # ✅ A tool deve ser capaz de identificar se há dados de moeda única ou usar o retorno de get_coin_details
   ```

3️⃣ **Gerar Insights (Tarefas)**
   ```python
   generate_tasks(capital=state["capital"])
   ```

4️⃣ **FINALIZAR: Gerar relatório executivo** (OBRIGATÓRIO)
   Você DEVE chamar esta ferramenta para completar a análise.
   Não passe argumentos.
   ```python
   generate_executive_report()
   ```

---

### 🅲 MODO FOLLOW-UP (mode="follow_up_analysis")
Siga este fluxo quando baseline_report_id estiver preenchido:

1️⃣ **Carregar análise baseline**
   ```python
   load_baseline_report(baseline_report_id=state["baseline_report_id"])
   ```

2️⃣ **Buscar dados atuais do mercado**
   ```python
   fetch_market_data()
   ```

3️⃣ **Comparar com baseline**
   ```python
   compare_with_baseline()
   ```

4️⃣ **Gerar tarefas baseadas na comparação**
   ```python
   generate_tasks(capital=state["capital"])
   ```

5️⃣ **FINALIZAR: Gerar relatório executivo** (OBRIGATÓRIO)
   Você DEVE chamar esta ferramenta para completar a análise.
   Não passe argumentos.
   ```python
   generate_executive_report()
   ```

---

**REGRAS ABSOLUTAS:**
✅ NÃO invente dados. Use as tools.
✅ NÃO passe JSONs gigantes como argumento. Confie no estado.
✅ Se uma tool falhar, tente recuperar ou notificar o erro.
✅ A ÚLTIMA AÇÃO DEVE SER `generate_executive_report`.
✅ Termine chamando `generate_executive_report` quando todos os dados estiverem prontos.
"""

# Ferramentas disponíveis
tools = [
    *get_market_tools(),
    *get_analysis_tools(),
    generate_executive_report,
    load_baseline_report,
    compare_with_baseline,
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
