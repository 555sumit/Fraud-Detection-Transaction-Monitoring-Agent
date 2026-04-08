from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from env import FraudDetectionEnv
from typing import Dict, Any

app = FastAPI()
env = FraudDetectionEnv()

@app.get("/")
def health_check():
    """Automated ping to Space URL - returns 200"""
    return {"status": "ok", "message": "OpenEnv Fraud Detection Active"}

@app.post("/reset")
async def reset(request: Request, task_level: str = "easy"):
    """Responds to reset() in a highly permissive way"""
    
    # 1. Safely check if the validator sent a JSON body instead of a query parameter
    try:
        body = await request.json()
        if isinstance(body, dict):
            # Check for common task variable names validators might use
            task_level = body.get("task_level", body.get("task", task_level))
    except Exception:
        pass # If no JSON body was sent, just fall back to the default "easy"

    if task_level not in ["easy", "medium", "hard"]:
        raise HTTPException(status_code=400, detail=f"Invalid task level: {task_level}")
        
    obs = env.reset(task_level=task_level)
    
    # 2. Return the observation DIRECTLY, not wrapped in a dictionary
    return obs

@app.post("/step")
def step(action: Dict[str, Any]):
    """Responds to step()"""
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    """Responds to state()"""
    return env.state()