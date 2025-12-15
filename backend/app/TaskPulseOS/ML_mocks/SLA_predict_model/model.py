def predict_sla(run_steps, workflow_def):
    """
    run_steps: list of RunStep dicts
    workflow_def: dict containing dag_json etc.
    """

    import pandas as pd
    import json
    from datetime import timedelta

    # ---------------------------------------------------------------------
    # Convert to DataFrame
    # ---------------------------------------------------------------------
    df = pd.DataFrame(run_steps)

    df["started_at"] = pd.to_datetime(df["started_at"])
    df["ended_at"] = pd.to_datetime(df["ended_at"])

    # Default SLA per step (3 min)
    df["sla_ms"] = df.get("sla_ms", pd.Series([180000] * len(df)))

    # fixed "now" for mock behavior
    now = pd.to_datetime("2025-01-01 10:06:00")

    # ---------------------------------------------------------------------
    # Step Metrics
    # ---------------------------------------------------------------------
    df = df.sort_values("started_at")

    # Latency
    df["latency_ms"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() * 1000

    # Queue time
    df["prev_end"] = df["ended_at"].shift(1)
    df["queue_time_ms"] = (df["started_at"] - df["prev_end"]).dt.total_seconds() * 1000
    df["queue_time_ms"] = df["queue_time_ms"].clip(lower=0)

    # SLA Remaining
    df["eta"] = df["started_at"] + pd.to_timedelta(df["sla_ms"], unit="ms")
    df["remaining_ms"] = (df["eta"] - now).dt.total_seconds() * 1000

    # Retry penalty
    df["retry_penalty"] = df["attempt"] * 15000

    # ---------------------------------------------------------------------
    # Critical Path
    # ---------------------------------------------------------------------
    df["critical_path_ms"] = df["remaining_ms"].rolling(window=2, min_periods=1).mean()

    # ---------------------------------------------------------------------
    # Global fallback rate
    # ---------------------------------------------------------------------
    #Filters the DataFrame to get only tasks started in the last 15 minutes
    recent = df[df["started_at"] >= (now - timedelta(minutes=15))]
    #Calculates percentage of recent tasks that failed, with protection against division by zero
    fallback_rate = (
        len(recent[recent["status"] == "failed"]) / len(recent)
        if len(recent) > 0 else 0
    )

    # ---------------------------------------------------------------------
    # Evaluate risk band (existing logic)
    # ---------------------------------------------------------------------
    latest = df.iloc[-1]#Get the Latest Row
    reasons = set()

    remaining = latest["remaining_ms"]
    critical_path = latest["critical_path_ms"]
    queue_time = latest["queue_time_ms"]
    retry_penalty = latest["retry_penalty"]

    if remaining < 0:
        risk = "breach"
        reasons.add("sla_breach")

    elif remaining < 60000 and critical_path > 90000:
        risk = "at_risk"
        reasons.add("sla_remaining_low")
        reasons.add("critical_path_heavy")

    elif queue_time > 20000:
        risk = "watch"
        reasons.add("long_queue_time")

    elif retry_penalty > 15000:
        risk = "watch"
        reasons.add("retry_inflation")

    else:
        risk = "ok"

    if fallback_rate > 0.15:
        reasons.add("high_fallback_rate")

    
    # Compute ML-style single risk score
    
    # Normalize values into a probability-like score
    risk_score_components = []

    # Remaining time pressure (low remaining → high risk)
    if latest["remaining_ms"] > 0:
        risk_score_components.append(
            min(1.0, critical_path / (latest["remaining_ms"] + 1))
        )
    else:
        risk_score_components.append(1.0)

    # Queue pressure
    risk_score_components.append(min(1.0, queue_time / 60000))

    # Retry penalty inflation
    risk_score_components.append(min(1.0, retry_penalty / 60000))

    # Fallback rate influence
    risk_score_components.append(min(1.0, fallback_rate * 3))

    # Final risk score = bounded mean of components
    risk_score = min(1.0, max(0.0, sum(risk_score_components) / len(risk_score_components)))

    
    # Actions from policy file
    with open("policies/sla_actions.json") as f:
        policies = json.load(f)
    actions = policies.get(risk, [])

    
    # Final Output
    return {         
        "risk": round(risk_score, 4), 
        "critical_path": critical_path,
        "fallback_rate": round(fallback_rate),
        "reasons": sorted(list(reasons)),
        "actions": actions
    }
