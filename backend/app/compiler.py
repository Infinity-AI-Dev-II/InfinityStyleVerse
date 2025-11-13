# backend/app/compiler.py
#   Provides minimal workflow compilation and step lookup utilities.
def compile_workflow_from_yaml(yaml_text: str) -> dict:
    """
    Minimal stub: parse YAML and return DAG nodes.
    Replace with your actual YAML compiler logic later.
    """
    import yaml
    dag = yaml.safe_load(yaml_text)
    # For simplicity, return nodes as a list
    return {"nodes": dag.get("steps", [])}

# Store step definitions globally for lookup
_STEP_DEFS = {}

def register_steps(step_defs):
    global _STEP_DEFS
    _STEP_DEFS = {step["id"]: step for step in step_defs}

def lookup_step(step_id: str):
    """
    Return the step definition by step_id
    """
    step = _STEP_DEFS.get(step_id)
    if not step:
        raise KeyError(f"Step {step_id} not found")
    return step
