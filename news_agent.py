"""
News Agent for Crypto News Analysis
Uses LangGraph with streaming for real-time UI updates
"""
from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from tools.news_tools import get_news_tools
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente do .env (incluindo LangSmith tracing)
load_dotenv()

# Agent state
class NewsAgentState(TypedDict):
    """State for news agent"""
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str
    mode: str  # "brief" | "search" | "chat"
    query: str
    news: List[dict]
    search_results: List[dict]
    sentiment_analysis: dict
    news_summary: dict
    summary_complete: bool


# Model configuration using OpenRouter
model = ChatOpenAI(
    model="anthropic/claude-4.5-sonnet",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", ""),
    max_tokens=8000,
    temperature=0.7
)

# System prompt for news agent
NEWS_SYSTEM_PROMPT = """Você é o Axos News, um assistente especializado em notícias de criptomoedas.

**VERIFIQUE O MODO DE OPERAÇÃO NO ESTADO:**
`mode` pode ser:
- "brief": Gerar resumo diário inteligente (AI Brief)
- "search": Buscar notícias por tópico específico
- "chat": Responder perguntas sobre notícias

---

### 📰 MODO BRIEF (mode="brief")
⚠️ **USE APENAS UMA FERRAMENTA:**

```python
result = generate_news_brief(limit=20)
```

Esta ferramenta FAZ TUDO automaticamente:
- Busca notícias
- Analisa sentimento
- Gera resumo

Após receber o resultado, gere uma resposta em português resumindo:
- Sentimento geral (bullish/bearish/neutral) com confiança
- Total de notícias analisadas
- Principais manchetes
- Tickers em destaque

---

### 🔍 MODO SEARCH (mode="search")
Busque notícias sobre um tópico específico:

```python
results = search_news_by_topic(topic=<query do usuário>, limit=10)
```

---

### 💬 MODO CHAT (mode="chat")
Responda perguntas sobre as notícias usando as ferramentas disponíveis.

---

**REGRAS:**
✅ No modo BRIEF, use APENAS `generate_news_brief` - NÃO encadeie outras ferramentas
✅ Responda em português do Brasil
✅ Seja conciso e objetivo
✅ NÃO invente dados
"""

# Available tools
tools = get_news_tools()

# Create deep agent
print("🔧 Creating news agent...")
news_agent = create_deep_agent(
    model=model,
    tools=tools,
    system_prompt=NEWS_SYSTEM_PROMPT,
    name="news_agent"
)

# Compile the graph
news_graph = news_agent

print("✅ News Agent compiled and ready!")
print(f"📋 Available tools: {[t.name for t in tools]}")

