# backend/app/seeds/seed_data.py
#   Provides seed functions to populate the database with initial data
#   for roles, permissions, users, and workflow definitions.
from datetime import datetime
from werkzeug.security import generate_password_hash

from backend.app.models.Heartbeat import Heartbeat
from backend.app.models.echo_models import Policy
from backend.app.models.run_steps import RunStep
from backend.app.models.runs import Run
from ..database import db, get_db_session
from ..models import User, Role, Permission
from ..models import WorkflowDef
import json
from datetime import datetime, timedelta
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DEFAULT_ROLES = [
    {"role_name": "admin"},
    {"role_name": "executive"},
    {"role_name": "brand_manager"},
]


DEFAULT_PERMISSIONS = {
    "admin": [
        {"system": "core", "module_access": "all"},
        {"system": "users", "module_access": "crud"},
        {"system": "roles", "module_access": "crud"},
        {"system": "infinitybrain", "module_access": "admin"},
    ],
    "executive": [
        {"system": "reports", "module_access": "read"},
        {"system": "dashboard", "module_access": "read"},
    ],
    "brand_manager": [
        {"system": "products", "module_access": "crud"},
        {"system": "campaigns", "module_access": "crud"},
    ],
}

# Sample users (passwords will be hashed via the model's property)
SAMPLE_USERS = [
    {
        "name": "Admin User",
        "email": "admin@example.com",
        "password": "Password123!",
        "role_name": "admin",
    },
    {
        "name": "Executive User",
        "email": "executive@example.com",
        "password": "Password123!",
        "role_name": "executive",
    },
    {
        "name": "Brand Manager User",
        "email": "brandmanager@example.com",
        "password": "Password123!",
        "role_name": "brand_manager",
    },
]

#heartbeat data
heartbeat_data = [
    {
        "worker_id": "w-12",
        "kind": "celery",
        "last_seen_at": datetime.utcnow(),
        "meta_json": {
            "load": 0.67,
            "queue_depth": 15,
            "version": "v1.0.2"
        }
    },
    {
        "worker_id": "w-17",
        "kind": "celery",
        "last_seen_at": datetime.utcnow() - timedelta(seconds=45),
        "meta_json": {
            "load": 0.12,
            "queue_depth": 0,
            "status": "stale"
        }
    },
    {
        "worker_id": "gpu-node-04",
        "kind": "inference_engine",
        "last_seen_at": datetime.utcnow(),
        "meta_json": {
            "gpu_utilization": 92,
            "memory_free": "4GB",
            "model_loaded": "llama-3-8b"
        }
    },
    {
        "worker_id": "ext-payment-gateway",
        "kind": "external_service",
        "last_seen_at": datetime.utcnow(),
        "meta_json": {
            "latency_p95": 240,
            "uptime_status": "healthy"
        }
    }
]
#run step data
now = datetime.utcnow()
ten_mins_ago = now - timedelta(minutes=10)
five_mins_ago = now - timedelta(minutes=5)

run_steps_data = [
    # 1. A Standard Succeeded Step
    # Based on the general "Runs & Step Tracking" [cite: 13]
    {
        "run_id": 1,
        "step_id": "validate_payload",
        "worker_id": "w-12",  # Worker ID from [cite: 82]
        "type": "validation",
        "status": "succeeded",
        "attempt": 1,
        "started_at": ten_mins_ago,
        "ended_at": ten_mins_ago + timedelta(seconds=2),
        "input_json": json.dumps({"customer_id": "C-998"}),
        "output_json": json.dumps({"valid": True}),
        "error_json": None
    },

    # 2. A Failed Step (Rate Limit Error)
    # Replicates the "Event Schema" example found on Page 3 [cite: 49, 55, 63]
    {
        "run_id": 2,
        "step_id": "price",  # Specific step ID from [cite: 55]
        "worker_id": "w-12",
        "type": "calculation",
        "status": "failed",  # Status from [cite: 56]
        "attempt": 3,        # Attempt count from [cite: 57]
        "started_at": five_mins_ago,
        "ended_at": five_mins_ago + timedelta(seconds=1),
        "input_json": json.dumps({"item_count": 5}),
        "output_json": None,
        # Error payload matches [cite: 63]
        "error_json": json.dumps({
            "error_code": "DOWNSTREAM_429",
            "message": "rate limit"
        })
    },

    # 3. A Running Step causing "SLA At Risk"
    # Based on Scenario A: "Step trend running long" [cite: 172]
    {
        "run_id": 3,
        "step_id": "model_inference",
        "worker_id": "gpu-node-01",
        "type": "ai_inference",
        "status": "running",
        "attempt": 1,
        "started_at": now - timedelta(seconds=45), # Running longer than expected
        "ended_at": None,
        "input_json": json.dumps({"prompt_tokens": 1024}),
        "output_json": None,
        "error_json": None
    },

    # 4. A Fallback Step
    # Based on Scenario C: "workflows use rules.simple_price" 
    {
        "run_id": 4,
        "step_id": "simple_price", # Alternate step used during fallback 
        "worker_id": "w-17",
        "type": "fallback_logic",
        "status": "succeeded",
        "attempt": 1,
        "started_at": now,
        "ended_at": now + timedelta(milliseconds=500),
        "input_json": json.dumps({"item_count": 5, "fallback_mode": True}),
        "output_json": json.dumps({"price": 19.99, "source": "static_rules"}),
        "error_json": None
    }
]
runs_data = [
    # 1. A Standard Successful Run (Completed)
    # Represents a healthy workflow for tenant "acme"[cite: 52].
    # Status: succeeded.
    {
        "workflow_id": 1,
        "version": "v1.2.0",
        "status": "succeeded",
        "tenant": "acme",
        "caller": "api_gateway",
        "inputs_json": json.dumps({
            "order_id": "ord-1024",
            "amount": 99.50,
            "currency": "USD"
        }),
        "started_at": now - timedelta(minutes=5),
        "ended_at": now - timedelta(minutes=4, seconds=30),  # 30s duration
        "created_at": now - timedelta(minutes=5),
        "updated_at": now - timedelta(minutes=4, seconds=30)
    },

    # 2. A Failed Run (Rate Limited)
    # Matches the scenario where a "price" step fails due to rate limits[cite: 63].
    # Status: failed[cite: 90].
    {
        "workflow_id": 1,
        "version": "v1.2.0",
        "status": "failed",
        "tenant": "globex",
        "caller": "web_checkout",
        "inputs_json": json.dumps({
            "order_id": "ord-1025",
            "items": ["widget-a", "widget-b"]
        }),
        "started_at": now - timedelta(minutes=10),
        "ended_at": now - timedelta(minutes=9, seconds=50),
        "created_at": now - timedelta(minutes=10),
        "updated_at": now - timedelta(minutes=9, seconds=50)
    },

    # 3. An Active Run (SLA At Risk)
    # This run is still "running" but has been active longer than usual.
    # This correlates to the "SLA At Risk" scenario[cite: 170, 172].
    {
        "workflow_id": 2,
        "version": "v2.0.0-beta",
        "status": "running",
        "tenant": "acme",
        "caller": "scheduler_cron",
        "inputs_json": json.dumps({
            "batch_id": "b-778",
            "mode": "deep_analysis"
        }),
        "started_at": now - timedelta(seconds=90), # Running for 90s (potentially long)
        "ended_at": None,
        "created_at": now - timedelta(seconds=90),
        "updated_at": now - timedelta(seconds=10) # Last heartbeat/update
    },

    # 4. A Compensating Run (Rolling Back)
    # The document explicitly lists "compensating" as a status[cite: 90].
    # This happens when a step fails and the workflow must undo previous steps.
    {
        "workflow_id": 1,
        "version": "v1.1.5",
        "status": "compensating",
        "tenant": "cyberdyne",
        "caller": "external_webhook",
        "inputs_json": json.dumps({
            "deployment_target": "us-east-1",
            "replicas": 5
        }),
        "started_at": now - timedelta(minutes=2),
        "ended_at": None,
        "created_at": now - timedelta(minutes=2),
        "updated_at": now - timedelta(seconds=5)
    }
]
policy_data = [
    {
           "task": "heartbeat",
           "version": "v1.0",
           "status": "active",
           "weights_blob": None,
           "config_json": {
               "interval_sec": 10,
               "miss_threshold": 3
           },
           "created_at": datetime.utcnow()
    },
    {
        "task": "sla_rule",
        "version": "v1.0",
        "status": "active",
        "weights_blob": None,
        "config_json": {
            "at_risk": 60,
            "critical_path": 50,
            "fallback_rate": 15
        },
        "created_at": datetime.utcnow()
    }
]
def createPolicyData():
    with get_db_session() as session:
        for policyData in policy_data:
            newObj = Policy(
                task = policyData["task"],
                version = policyData["version"],
                status = policyData["status"],
                weights_blob = policyData["weights_blob"],
                config_json = policyData["config_json"],
                created_at = policyData["created_at"]
            )
            session.add(newObj)
            session.commit()
    logger.info("Policy data seeded successfully.")        
def createRunData():
    with get_db_session() as session:
        for runData in runs_data:
            runObj = Run(
                workflow_id=runData["workflow_id"],
                version=runData["version"],
                status=runData["status"],
                tenant=runData["tenant"],
                caller=runData["caller"],
                inputs_json=runData["inputs_json"],
                started_at=runData["started_at"],
                ended_at=runData["ended_at"],
                created_at=runData["created_at"],
                updated_at=runData["updated_at"]
            )
            session.add(runObj)
            session.commit()
    logger.info("Run data seeded successfully.")
def createHeartbeatData():
    with get_db_session() as session:
        for data in heartbeat_data:
            newObj = Heartbeat(
                worker_id = data["worker_id"],
                kind = data["kind"],
                last_seen_at = data["last_seen_at"],
                meta_json = data["meta_json"]
            )
            session.add(newObj)
            session.commit()
    logger.info("Heartbeat data seeded successfully.")      
def createRunStepData():
    with get_db_session() as session:
        for stepData in run_steps_data:
            runStepObj = RunStep(
                run_id=stepData["run_id"],
                step_id=stepData["step_id"],
                worker_id=stepData["worker_id"],
                type=stepData["type"],
                status=stepData["status"],
                attempt=stepData["attempt"],
                started_at=stepData["started_at"],
                ended_at=stepData["ended_at"],
                input_json=stepData["input_json"],
                output_json=stepData["output_json"],
                error_json=stepData["error_json"]
            )
            session.add(runStepObj)
            session.commit()
    logger.info("Run steps data seeded successfully.")              
def get_or_create_role(role_name: str) -> Role:
    role = Role.query.filter_by(role_name=role_name).first()
    if role:
        return role
    role = Role(role_name=role_name)
    db.session.add(role)
    db.session.flush()  # get role.id
    return role


def ensure_permissions_for_role(role: Role, perms: list[dict]):
    """Idempotent: only adds missing permissions."""
    existing = {(p.system, p.module_access) for p in role.permissions}
    for p in perms:
        key = (p["system"], p["module_access"])
        if key in existing:
            continue
        db.session.add(Permission(role_id=role.id, **p))


def seed_roles_and_permissions() -> None:
    for r in DEFAULT_ROLES:
        role = get_or_create_role(r["role_name"])
        ensure_permissions_for_role(role, DEFAULT_PERMISSIONS.get(role.role_name, []))
    db.session.commit()


def get_or_create_user(name: str, email: str, password: str, role_name: str) -> User:
    user = User.query.filter_by(email=email).first()
    if user:
       
        role = Role.query.filter_by(role_name=role_name).first()
        if role and user.role_id != role.id:
            user.role = role
            db.session.commit()
        return user

    role = Role.query.filter_by(role_name=role_name).first()
    if not role:
        role = get_or_create_role(role_name)

    user = User(
        name=name,
        email=email,
        role=role,
        last_login=None,
        status="Inactive",
    )
    
    user.password = password
    db.session.add(user)
    db.session.commit()
    return user


def seed_users() -> None:
    for u in SAMPLE_USERS:
        get_or_create_user(
            name=u["name"],
            email=u["email"],
            password=u["password"],
            role_name=u["role_name"],
        )


def run_seed() -> None:
    """Main entry point for CLI command."""
    seed_roles_and_permissions()
    seed_users()
    print(" Seed complete: roles, permissions, and sample users inserted (idempotent).")

SAMPLE_WORKFLOWS = [
    {
        "name": "Default Workflow",
        "version": "1.0",
        "dsl_yaml": json.dumps({
            "nodes": {
                "start": {
                    "id": "start",
                    "type": "task",
                    "action": "noop",
                    "incoming": []
                }
            },
            "edges": []
        }),
        "created_by": "seed",
    },
    {
        "name": "Fail/Success Workflow",
        "version": "1.0",
        "dsl_yaml": json.dumps({
            "nodes": {
                "start": {
                    "id": "start",
                    "type": "task",
                    "action": "noop",
                    "incoming": []
                }
            },
            "edges": []
        }),
        "created_by": "seed",
    },
]

def get_or_create_workflow(name: str, version: str, dsl_yaml: str, created_by: str) -> WorkflowDef:
    wf = WorkflowDef.query.filter_by(name=name, version=version).first()
    if wf:
        return wf
    wf = WorkflowDef(name=name, version=version, dsl_yaml=dsl_yaml, created_by=created_by)
    db.session.add(wf)
    db.session.commit()
    return wf

def seed_workflows() -> None:
    for wf in SAMPLE_WORKFLOWS:
        get_or_create_workflow(
            name=wf["name"],
            version=wf["version"],
            dsl_yaml=wf["dsl_yaml"],
            created_by=wf["created_by"]
        )
    print(f" Seeded {len(SAMPLE_WORKFLOWS)} workflows.")

def run_seed() -> None:
    seed_roles_and_permissions()
    seed_users()
    createPolicyData()
    seed_workflows() 
    createHeartbeatData()
    createRunData()
    createRunStepData()
    logger.info(" Seed complete: roles, permissions, users, and workflows inserted (idempotent).")


