#check the health od the worker based on the heartbeats and the policy
def CheckWorkerHealth(policy_json, oldHB, newHB):
    try:
        old = oldHB.last_seen_at
        new = newHB.last_seen_at
        difference = (new - old).total_seconds()
        generalLimit = policy_json["interval_sec"] * policy_json["miss_threshold"]
        if difference < generalLimit:
            return {"Status": "Healthy", "Message": "Worker is healthy"}
        else:
            return {"Status": "Unhealthy", "Message": "Worker is unhealthy"}
    except Exception as e:
        return {"Status": "Error", "Message": str(e)}
    
    
    