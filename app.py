from fastapi import FastAPI, HTTPException
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
def reset(task_level: str = "easy"):
    """Responds to reset()"""
    if task_level not in ["easy", "medium", "hard"]:
        raise HTTPException(status_code=400, detail="Invalid task level")
    obs = env.reset(task_level=task_level)
    return {"observation": obs}

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