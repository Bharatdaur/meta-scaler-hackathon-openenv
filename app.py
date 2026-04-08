from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from env import FinOpsEnv
from models import Action
import os

app = FastAPI(title="FinOps AI Environment API", description="OpenEnv server for Cloud Cost Optimization")
env = FinOpsEnv()

class ResetRequest(BaseModel):
    task_id: str = "easy"

@app.get("/")
def read_root():
    return {"status": "ok", "message": "FinOps OpenEnv API is running on Hugging Face Spaces!"}

@app.post("/reset")
def reset_environment(req: ResetRequest):
    try:
        obs = env.reset(req.task_id)
        return obs
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step")
def step_environment(action: Action):
    if env._state is None:
        raise HTTPException(status_code=400, detail="Environment not reset. Call /reset first.")
    
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def get_state():
    if env._state is None:
         raise HTTPException(status_code=400, detail="Environment not reset. Call /reset first.")
    return env.state()

# Provide an entry point for HF Spaces
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
