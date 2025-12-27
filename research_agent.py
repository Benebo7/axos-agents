"""
Research Agent para interpretar linguagem natural e criar automações de análise
"""
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

# Estado do agente
class ResearchAgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    automation: dict  # Resultado estruturado da automação

# Modelo de saída estruturada
class AutomationConfig(BaseModel):
    """Configuração de automação de análise"""
    coin_id: str = Field(description="ID da moeda (ex: bitcoin, ethereum)")
    prompt: str = Field(description="Prompt que será enviado ao agente de análise")
    frequency: str = Field(description="Frequência: daily, weekly, biweekly, monthly, once")
    repeat: bool = Field(description="Se deve repetir ou executar apenas uma vez")
    day_of_week: int | None = Field(default=None, description="Dia da semana (0-6, 0=segunda) para weekly")
    day_of_month: int | None = Field(default=None, description="Dia do mês (1-31) para monthly")
    time_of_day: str = Field(default="09:00", description="Horário de execução (HH:MM)")

RESEARCH_SYSTEM_PROMPT = """Você é um assistente especializado em criar automações de análise de criptomoedas.

Sua tarefa é interpretar o pedido do usuário e gerar uma configuração estruturada de automação.

**Exemplos de interpretação:**

Usuário: "Analise o RSI do Bitcoin toda segunda-feira"
→ coin_id: "bitcoin"
→ prompt: "Faça uma análise completa do indicador RSI do Bitcoin nos últimos 7 dias"
→ frequency: "weekly"
→ repeat: true
→ day_of_week: 0 (segunda)
→ time_of_day: "09:00"

Usuário: "Quero análise de Ethereum a cada 2 semanas"
→ coin_id: "ethereum"
→ prompt: "Faça uma análise completa do Ethereum nos últimos 14 dias"
→ frequency: "biweekly"
→ repeat: true
→ day_of_week: 0
→ time_of_day: "09:00"

Usuário: "Analise Solana apenas uma vez amanhã"
→ coin_id: "solana"
→ prompt: "Faça uma análise completa do Solana"
→ frequency: "once"
→ repeat: false
→ time_of_day: "09:00"

**Regras:**
- Sempre gere um prompt claro e completo para o agente de análise
- Se o usuário não especificar frequência, use "weekly"
- Se não especificar dia, use segunda-feira (0)
- Se não especificar horário, use 09:00
- O prompt deve incluir período de análise apropriado (7 dias para weekly, 14 para biweekly, etc)
"""

model = ChatOpenAI(
    model="anthropic/claude-4.5-sonnet",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", ""),
    max_tokens=2000,
    temperature=0.3
).with_structured_output(AutomationConfig)

def create_automation(state: ResearchAgentState):
    """Interpreta o pedido do usuário e cria automação"""
    system_message = {"role": "system", "content": RESEARCH_SYSTEM_PROMPT}
    messages = [system_message] + state["messages"]
    
    automation_config = model.invoke(messages)
    
    return {
        "automation": automation_config.model_dump(),
        "messages": [{"role": "assistant", "content": f"Automação criada: {automation_config.coin_id} - {automation_config.frequency}"}]
    }

# Construir grafo
workflow = StateGraph(ResearchAgentState)
workflow.add_node("create_automation", create_automation)
workflow.add_edge(START, "create_automation")
workflow.add_edge("create_automation", END)

research_graph = workflow.compile()

print("✅ Research Agent compilado e pronto!")

