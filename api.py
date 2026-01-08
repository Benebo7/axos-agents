"""
API FastAPI para gerenciar automações de análise
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scheduler import scheduler
from execution_limiter import get_stats
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AutomationRequest(BaseModel):
    user_id: str
    coin_id: str
    prompt: str
    frequency: str
    repeat: bool
    day_of_week: int | None = None
    day_of_month: int | None = None
    time_of_day: str = "09:00"

@app.post("/automations")
def create_automation(req: AutomationRequest):
    automation_id = str(uuid.uuid4())
    config = req.model_dump()
    config["id"] = automation_id
    
    scheduler.add_automation(automation_id, config)
    
    return {"automation_id": automation_id, "status": "created"}

@app.delete("/automations/{automation_id}")
def delete_automation(automation_id: str):
    scheduler.remove_automation(automation_id)
    return {"status": "deleted"}

@app.post("/automations/{automation_id}/pause")
def pause_automation(automation_id: str):
    scheduler.pause_automation(automation_id)
    return {"status": "paused"}

@app.post("/automations/{automation_id}/resume")
def resume_automation(automation_id: str):
    scheduler.resume_automation(automation_id)
    return {"status": "resumed"}

@app.get("/automations")
def list_automations():
    return scheduler.automations

@app.get("/automations/{automation_id}/results")
def get_results(automation_id: str):
    import json
    from pathlib import Path
    
    results_file = Path("automation_results.json")
    if results_file.exists():
        with open(results_file, 'r') as f:
            all_results = json.load(f)
            return all_results.get(automation_id, [])
    return []

@app.get("/health")
def health_check():
    """Health check + status do semaphore de concorrência."""
    stats = get_stats()
    return {
        "status": "healthy",
        "concurrency": stats,
    }


@app.on_event("startup")
def startup_event():
    scheduler.start()

@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

