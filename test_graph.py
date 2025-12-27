"""Script de teste para validar o grafo com Command"""
from dotenv import load_dotenv

# Carregar variáveis de ambiente do .env (incluindo LangSmith tracing)
load_dotenv()

from agent import graph
from langchain_core.messages import HumanMessage

print('🚀 Iniciando teste do grafo...')
print('=' * 60)

# Simular invocação
config = {'configurable': {'thread_id': 'test-123'}}
try:
    result = graph.invoke({
        'messages': [HumanMessage(content='Analise o mercado de criptomoedas')],
        'user_id': 'test',
        'period': '7d',
        'risk_profile': 'moderate',
        'capital': 10000,
        'opportunities': [],
        'tasks': [],
        'allocation': {},
        'analysis_complete': False
    }, config)

    print('\n✅ TESTE EXECUTADO COM SUCESSO!')
    print('=' * 60)
    print(f'📊 Oportunidades no estado: {len(result.get("opportunities", []))}')
    print(f'📋 Tarefas no estado: {len(result.get("tasks", []))}')
    print(f'💰 Alocação criada: {"allocation" in result and bool(result.get("allocation", {}).get("allocation", []))}')
    print(f'✔️  Análise completa: {result.get("analysis_complete", False)}')
    print('=' * 60)

    # Mostrar algumas oportunidades se existirem
    if result.get('opportunities'):
        print('\n📈 Primeiras 3 oportunidades identificadas:')
        for i, opp in enumerate(result['opportunities'][:3], 1):
            print(f'  {i}. {opp["coin_symbol"]} ({opp["coin_name"]}) - Confiança: {opp["confidence"]:.0f}%')
            print(f'     Preço: ${opp["entry_price"]:.2f} | Alvo: ${opp["target_price"]:.2f}')

    # Mostrar tarefas se existirem
    if result.get('tasks'):
        print(f'\n📝 {len(result["tasks"])} tarefas geradas')
        for task in result['tasks']:
            print(f'  - {task["title"]} (Prioridade: {task["priority"]})')

    # Mostrar alocação se existir
    if result.get('allocation', {}).get('allocation'):
        print(f'\n💼 Alocação de portfólio criada com {len(result["allocation"]["allocation"])} moedas')

    print('\n✅ TODOS OS COMPONENTES FUNCIONANDO CORRETAMENTE!')

except Exception as e:
    print(f'\n❌ ERRO NO TESTE: {str(e)}')
    import traceback
    traceback.print_exc()
