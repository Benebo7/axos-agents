"""
Sistema de agendamento de automações com APScheduler
"""
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from langchain_core.messages import HumanMessage
from agent import graph as analysis_graph

# Arquivo de persistência de automações
AUTOMATIONS_FILE = Path("automations.json")
RESULTS_FILE = Path("automation_results.json")

class AutomationScheduler:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.automations = self._load_automations()
        
    def _load_automations(self):
        """Carrega automações do arquivo JSON"""
        if AUTOMATIONS_FILE.exists():
            with open(AUTOMATIONS_FILE, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_automations(self):
        """Salva automações no arquivo JSON"""
        with open(AUTOMATIONS_FILE, 'w') as f:
            json.dump(self.automations, f, indent=2)
    
    def _save_result(self, automation_id: str, result: dict):
        """Salva resultado de uma execução"""
        results = {}
        if RESULTS_FILE.exists():
            with open(RESULTS_FILE, 'r') as f:
                results = json.load(f)
        
        if automation_id not in results:
            results[automation_id] = []
        
        results[automation_id].append({
            "timestamp": datetime.now().isoformat(),
            "result": result
        })
        
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
    
    def _execute_automation(self, automation_id: str):
        """Executa uma automação"""
        automation = self.automations.get(automation_id)
        if not automation:
            return
        
        try:
            # Executar análise com o prompt da automação
            result = analysis_graph.invoke({
                "messages": [HumanMessage(content=automation["prompt"])],
                "user_id": automation["user_id"],
                "period": "7d",  # Padrão
                "risk_profile": "moderate",
                "capital": 0,
                "mode": "single_coin_analysis",
                "coin_id": automation["coin_id"]
            })
            
            # Salvar resultado
            self._save_result(automation_id, {
                "opportunities": result.get("opportunities", []),
                "tasks": result.get("tasks", []),
                "executive_report": result.get("executive_report", {})
            })
            
            print(f"✅ Automação {automation_id} executada com sucesso")
            
        except Exception as e:
            print(f"❌ Erro ao executar automação {automation_id}: {e}")
            self._save_result(automation_id, {"error": str(e)})
    
    def add_automation(self, automation_id: str, config: dict):
        """Adiciona uma nova automação"""
        self.automations[automation_id] = config
        self._save_automations()
        
        # Criar trigger baseado na frequência
        if config["frequency"] == "once":
            # Executar uma vez no horário especificado
            run_time = datetime.now() + timedelta(minutes=1)  # Simplificado
            trigger = DateTrigger(run_date=run_time)
        elif config["frequency"] == "daily":
            hour, minute = config["time_of_day"].split(":")
            trigger = CronTrigger(hour=int(hour), minute=int(minute))
        elif config["frequency"] == "weekly":
            hour, minute = config["time_of_day"].split(":")
            trigger = CronTrigger(
                day_of_week=config["day_of_week"],
                hour=int(hour),
                minute=int(minute)
            )
        elif config["frequency"] == "biweekly":
            # A cada 2 semanas (simplificado - executar semanalmente e filtrar)
            hour, minute = config["time_of_day"].split(":")
            trigger = CronTrigger(
                day_of_week=config["day_of_week"],
                hour=int(hour),
                minute=int(minute),
                week="*/2"
            )
        elif config["frequency"] == "monthly":
            hour, minute = config["time_of_day"].split(":")
            trigger = CronTrigger(
                day=config["day_of_month"],
                hour=int(hour),
                minute=int(minute)
            )
        
        # Adicionar job ao scheduler
        self.scheduler.add_job(
            self._execute_automation,
            trigger=trigger,
            args=[automation_id],
            id=automation_id,
            replace_existing=True
        )
        
        print(f"✅ Automação {automation_id} agendada: {config['frequency']}")
    
    def remove_automation(self, automation_id: str):
        """Remove uma automação"""
        if automation_id in self.automations:
            del self.automations[automation_id]
            self._save_automations()
            
            try:
                self.scheduler.remove_job(automation_id)
            except:
                pass
            
            print(f"✅ Automação {automation_id} removida")
    
    def pause_automation(self, automation_id: str):
        """Pausa uma automação"""
        try:
            self.scheduler.pause_job(automation_id)
            if automation_id in self.automations:
                self.automations[automation_id]["paused"] = True
                self._save_automations()
            print(f"⏸️ Automação {automation_id} pausada")
        except Exception as e:
            print(f"❌ Erro ao pausar: {e}")
    
    def resume_automation(self, automation_id: str):
        """Retoma uma automação"""
        try:
            self.scheduler.resume_job(automation_id)
            if automation_id in self.automations:
                self.automations[automation_id]["paused"] = False
                self._save_automations()
            print(f"▶️ Automação {automation_id} retomada")
        except Exception as e:
            print(f"❌ Erro ao retomar: {e}")
    
    def start(self):
        """Inicia o scheduler"""
        # Recarregar automações existentes
        for automation_id, config in self.automations.items():
            if not config.get("paused", False):
                self.add_automation(automation_id, config)
        
        self.scheduler.start()
        print("🚀 Scheduler iniciado")
    
    def shutdown(self):
        """Para o scheduler"""
        self.scheduler.shutdown()
        print("🛑 Scheduler parado")

# Instância global
scheduler = AutomationScheduler()

