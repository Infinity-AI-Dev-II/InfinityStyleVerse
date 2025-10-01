# services/planner.py
import json
try:
    import yaml
except Exception:
    yaml = None

# services/planner.py
def parse_dsl(dsl):
    """
    Convert DSL (YAML/JSON) into DAG steps format compatible with start_run_from_dag.
    Always returns: {"steps": [...]} or raises exception on invalid DSL.
    """

    # If DSL is a string, try to parse as JSON
    if isinstance(dsl, str):
        import json, yaml
        try:
            try:
                dsl_obj = json.loads(dsl)
            except json.JSONDecodeError:
                dsl_obj = yaml.safe_load(dsl)
        except Exception as e:
            raise ValueError(f"Cannot parse DSL string: {e}")
    elif isinstance(dsl, dict):
        dsl_obj = dsl
    else:
        raise ValueError(f"Invalid DSL type: {type(dsl)}")

    # If DSL already contains 'steps', just return
    if "steps" in dsl_obj:
        return dsl_obj

    # If DSL contains nodes/edges (old format), convert to steps
    nodes = dsl_obj.get("nodes", {})
    edges = dsl_obj.get("edges", [])
    if nodes:
        steps = []
        for node_id, node_meta in nodes.items():
            depends_on = [a for a, b in edges if b == node_id]
            steps.append({
                "id": node_id,
                "action": node_meta.get("action"),
                "depends_on": depends_on
            })
        return {"steps": steps}

    # Fallback: if DSL is just a list of steps
    if isinstance(dsl_obj, list):
        return {"steps": dsl_obj}

    raise ValueError(f"Cannot convert DSL to steps: {dsl_obj}")
