"""
Deep Agent para Chat com o usuário
"""
from typing import List
from typing_extensions import NotRequired, TypedDict
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from tools.market_tools import get_market_tools
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente do .env (incluindo LangSmith tracing)
load_dotenv()


# Schema de contexto customizado para o chat
class ChatContext(TypedDict):
    """Campos extras que o BFF envia além das messages"""
    current_message: NotRequired[str]          # Mensagem atual do usuário
    mentions: NotRequired[List[dict]]          # Dados da UI anexados pelo usuário


# Configuração do modelo usando OpenRouter (igual ao agent.py)
model = ChatOpenAI(
    model="anthropic/claude-4.5-sonnet",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY", ""),
    max_tokens=8000,
    temperature=0.7
)

CHAT_SYSTEM_PROMPT = """Você é o Axos, um assistente especializado em criptomoedas.
Seu objetivo é ajudar o usuário com informações sobre o mercado, preços, tendências e análises.

Você tem acesso a ferramentas para buscar dados em tempo real do mercado:
- fetch_market_data: para buscar uma lista de moedas e visão geral.
- get_coin_details: para buscar detalhes específicos de uma moeda.

USE essas ferramentas sempre que o usuário perguntar sobre preços atuais, tendências ou detalhes de um ativo.
NÃO invente dados. Se não souber, use a ferramenta ou diga que não sabe.

Seja direto, útil e educado. Responda em português do Brasil.
"""

# Ferramentas disponíveis
tools = get_market_tools()

# Criar deep agent para chat
print("🔧 Criando chat agent...")
chat_agent = create_deep_agent(
    model=model,
    tools=tools,
    system_prompt=CHAT_SYSTEM_PROMPT,
    name="chat_agent",
    context_schema=ChatContext
)

# Compilar o grafo
chat_graph = chat_agent

print("✅ Chat Agent compilado e pronto!")
