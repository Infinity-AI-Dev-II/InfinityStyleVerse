from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
from bandits import ThompsonSampling, LinUCB, LogisticUCB
import numpy as np

app = FastAPI(title="Bandit Microservice")

# Initializing the bandits (3 arms and context dimension - 5)
thompson = ThompsonSampling(n_arms=3)
linucb = LinUCB(n_arms=3, dim=5, alpha=1.0)
logisticucb = LogisticUCB(n_arms=3, dim=5)

# Request Models 
class SelectArmRequest(BaseModel):
    task: str
    context: Optional[List[float]] = None

class UpdateRequest(BaseModel):
    chosen_arm: int = Field(ge=0)
    reward: float
    task: str
    context: Optional[List[float]] = None
    
@app.post("/select_arm")
async def select_arm(req: SelectArmRequest):
    task = req.task
    context = np.array(req.context) if req.context else np.zeros(5)

    if task == "outfit_ranking":
        response = thompson.select_arm()
        return {
            "task": task,
            "selected_arm": response["arm"],
            "method": "ThompsonSampling",
            "propensities": response["propensities"],
            "confidence_bounds": response["confidence_bounds"],
            "uncertainties": response.get("uncertainties")  # None for ThompsonSampling
        }

    elif task == "price_nudge":
        response = linucb.select_arm(context)
        return {
            "task": task,
            "selected_arm": response["arm"],
            "method": "LinUCB",
            "context_used": req.context,
            "propensities": response["propensities"],
            "confidence_bounds": response["confidence_bounds"],
            "uncertainties": response["uncertainties"]
        }

    else:
        response = logisticucb.select_arm(context)
        return {
            "task": task,
            "selected_arm": response["arm"],
            "method": "LogisticUCB",
            "context_used": req.context,
            "propensities": response["propensities"],
            "confidence_bounds": response["confidence_bounds"],
            "uncertainties": response["uncertainties"]
        }

@app.post("/update")
async def update(req: UpdateRequest):
    chosen_arm = req.chosen_arm
    reward = req.reward
    task = req.task
    context = np.array(req.context) if req.context else np.zeros(5)

    if task == "outfit_ranking":
        thompson.update(chosen_arm, reward)
    elif task == "price_nudge":
        linucb.update(chosen_arm, reward, context)
    else:
        logisticucb.update(chosen_arm, reward, context)

    return {"status": "updated", "task": task, "chosen_arm": chosen_arm, "reward": reward}