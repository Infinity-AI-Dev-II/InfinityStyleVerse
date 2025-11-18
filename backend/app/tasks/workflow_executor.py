from backend.app.event_metrics import track_step_execution, track_retry, track_compensation

def execute_step(step, workflow_name, run_id):
    step_id = step.id

    def step_logic():
        # your step logic here
        if step.type == "task" and step.payload.get("should_fail"):
            raise Exception("Simulated failure")
        return "ok"
    #Executes a workflow step and updates metrics + tracing.
    return track_step_execution(workflow_name, step_id, step_logic)
