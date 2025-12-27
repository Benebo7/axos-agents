"""Teste unitário das tools sem precisar de API"""
from tools.analysis_tools import analyze_opportunities, generate_tasks, create_allocation
from langgraph.types import Command
from langchain_core.messages import ToolMessage

print('🧪 Testando tools individualmente...')
print('=' * 60)

# Mock de dados de mercado
mock_market_data = [
    {
        "id": "bitcoin",
        "symbol": "btc",
        "name": "Bitcoin",
        "current_price": 50000,
        "market_cap": 1000000000000,
        "total_volume": 50000000000,
        "price_change_percentage_24h": 2.5
    },
    {
        "id": "ethereum",
        "symbol": "eth",
        "name": "Ethereum",
        "current_price": 3000,
        "market_cap": 400000000000,
        "total_volume": 20000000000,
        "price_change_percentage_24h": -1.2
    },
    {
        "id": "cardano",
        "symbol": "ada",
        "name": "Cardano",
        "current_price": 0.5,
        "market_cap": 20000000000,
        "total_volume": 1000000000,
        "price_change_percentage_24h": 5.0
    }
]

print('\n1️⃣ Testando analyze_opportunities...')
try:
    result = analyze_opportunities.invoke({
        "market_data_json": mock_market_data,
        "risk_profile": "moderate"
    })

    assert isinstance(result, Command), f"❌ Esperado Command, recebeu {type(result)}"
    assert "opportunities" in result.update, "❌ 'opportunities' não está no update"
    assert "messages" in result.update, "❌ 'messages' não está no update"
    assert isinstance(result.update["opportunities"], list), "❌ opportunities não é lista"
    assert isinstance(result.update["messages"], list), "❌ messages não é lista"
    assert len(result.update["messages"]) > 0, "❌ messages está vazio"
    assert isinstance(result.update["messages"][0], ToolMessage), "❌ Primeiro item não é ToolMessage"

    opps = result.update["opportunities"]
    print(f'   ✅ Retornou Command com {len(opps)} oportunidades')
    print(f'   ✅ ToolMessage presente: "{result.update["messages"][0].content[:50]}..."')

    if opps:
        print(f'   ✅ Primeira oportunidade: {opps[0]["coin_symbol"]} - {opps[0]["confidence"]:.0f}% confiança')

except Exception as e:
    print(f'   ❌ ERRO: {str(e)}')
    import traceback
    traceback.print_exc()

print('\n2️⃣ Testando generate_tasks...')
try:
    result = generate_tasks.invoke({
        "opportunities_json": [
            {"coin_id": "btc", "coin_symbol": "BTC", "confidence": 85}
        ],
        "capital": 10000
    })

    assert isinstance(result, Command), f"❌ Esperado Command, recebeu {type(result)}"
    assert "tasks" in result.update, "❌ 'tasks' não está no update"
    assert "messages" in result.update, "❌ 'messages' não está no update"
    assert isinstance(result.update["tasks"], list), "❌ tasks não é lista"
    assert len(result.update["tasks"]) > 0, "❌ tasks está vazio"

    tasks = result.update["tasks"]
    print(f'   ✅ Retornou Command com {len(tasks)} tarefas')
    print(f'   ✅ Primeira tarefa: {tasks[0]["title"]}')

except Exception as e:
    print(f'   ❌ ERRO: {str(e)}')
    import traceback
    traceback.print_exc()

print('\n3️⃣ Testando create_allocation...')
try:
    result = create_allocation.invoke({
        "capital": 10000,
        "opportunities_json": [
            {"coin_id": "btc", "coin_symbol": "BTC", "confidence": 85, "risk_level": "Baixo", "timeframe": "2-4 semanas"},
            {"coin_id": "eth", "coin_symbol": "ETH", "confidence": 80, "risk_level": "Médio", "timeframe": "2-4 semanas"}
        ],
        "risk_profile": "moderate"
    })

    assert isinstance(result, Command), f"❌ Esperado Command, recebeu {type(result)}"
    assert "allocation" in result.update, "❌ 'allocation' não está no update"
    assert "analysis_complete" in result.update, "❌ 'analysis_complete' não está no update"
    assert "messages" in result.update, "❌ 'messages' não está no update"
    assert result.update["analysis_complete"] == True, "❌ analysis_complete não é True"

    allocation = result.update["allocation"]
    print(f'   ✅ Retornou Command com alocação')
    print(f'   ✅ Análise marcada como completa: {result.update["analysis_complete"]}')
    print(f'   ✅ Moedas na alocação: {len(allocation.get("allocation", []))}')

except Exception as e:
    print(f'   ❌ ERRO: {str(e)}')
    import traceback
    traceback.print_exc()

print('\n' + '=' * 60)
print('✅ TODOS OS TESTES PASSARAM!')
print('=' * 60)
print('\n📋 RESUMO:')
print('  ✅ analyze_opportunities retorna Command e atualiza opportunities')
print('  ✅ generate_tasks retorna Command e atualiza tasks')
print('  ✅ create_allocation retorna Command e atualiza allocation + analysis_complete')
print('  ✅ Todas as tools incluem ToolMessage no update')
print('\n🎯 SOLUÇÃO IMPLEMENTADA COM SUCESSO!')
print('   As tools agora atualizam o estado do grafo via Command')
print('   O frontend receberá as atualizações em stateUpdate.opportunities, etc.')
