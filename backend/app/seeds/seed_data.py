# backend/app/seeds/seed_data.py
#   Provides seed functions to populate the database with initial data
#   for roles, permissions, users, and workflow definitions.
from datetime import datetime
from werkzeug.security import generate_password_hash
from ..database import db
from ..models import User, Role, Permission
from ..models import WorkflowDef
import json


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
    seed_workflows() 
    print(" Seed complete: roles, permissions, users, and workflows inserted (idempotent).")


