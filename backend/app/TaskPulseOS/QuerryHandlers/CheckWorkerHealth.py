#check the health od the worker based on the heartbeats and the policy
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def CheckWorkerHealth(policy_json, oldHB, newHB):
    try:
        # old =oldHB.last_seen_at
        # new = newHB.last_seen_at
        # logger.info(f"Type of old: {type(old)}")
        # logger.info(f"Value of old: {old}")
        # logger.info(f"Type of new: {type(new)}")
        # logger.info(f"Value of new: {new}")
        
        # # Convert string to datetime
        # if isinstance(new, str):
        #     # Handle ISO format with 'Z' timezone
        #     if new.endswith('Z'):
        #         # Python 3.11+ supports 'Z' directly, otherwise:
        #         new = new.replace('Z', '+00:00')
        #     new = datetime.fromisoformat(new)
        
        # difference = (new - old).total_seconds()
        
        old = oldHB.last_seen_at  # naive datetime
        
        # Convert new from string to datetime
        if isinstance(newHB.last_seen_at, str):
            new_str = newHB.last_seen_at
            if new_str.endswith('Z'):
                new_str = new_str.replace('Z', '+00:00')
            new = datetime.fromisoformat(new_str)
        else:
            new = newHB.last_seen_at
        
        # Make aware datetime naive by removing timezone info
        if new.tzinfo is not None: 
            new = new.replace(tzinfo=None)
        
        # Now both are naive
        difference = (new - old).total_seconds()
        logger.info("difference number: " + str(difference) + "\n new: " + str(new) + "\nold: " + str(old))
        generalLimit = policy_json["interval_sec"] * policy_json["miss_threshold"]
        if difference < generalLimit:
            return {"Status": "Healthy", "Message": "Worker is healthy"}
        else:
            return {"Status": "Unhealthy", "Message": "Worker is unhealthy"}
    except Exception as e:
        return {"Status": "Error in CheckWorkerHealth", "Message": str(e)}
    
    
    