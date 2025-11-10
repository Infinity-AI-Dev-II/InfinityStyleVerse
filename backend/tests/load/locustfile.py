# backend/tests/load/locustfile.py
#   Load testing script using Locust to simulate user interactions with the workflow API.
from locust import HttpUser, task, between
import random
import string
import json

API_HOST = "http://api:8000"  # your API host

def random_string(length=6):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

class WorkflowUser(HttpUser):
    wait_time = between(1, 3)
    host = API_HOST
    valid_workflow_ids = []
    token = None
    run_ids = []   # ✅ keep track of created run IDs

    def on_start(self):
        # Step 1: Authenticate and get JWT token
        credentials = {
        "email": "admin@example.com",
        "password": "Password123!"
    }


# ensure all API calls include headers=self.auth_header()
 # replace with seed user
        try:
            resp = self.client.post("/auth/login", json=credentials)
            if resp.status_code == 200:
                self.token = resp.json().get("access_token")
                print(f"Logged in successfully. Token: {self.token}")
            else:
                print(f"Login failed: {resp.status_code} | {resp.text}")
        except Exception as e:
            print(f"Login exception: {e}")

        # Step 2: Fetch valid workflow IDs
        try:
            resp = self.client.get("/flow/workflow_defs", headers=self.auth_header())
            if resp.status_code == 200:
                workflows = resp.json()
                self.valid_workflow_ids = [wf["id"] for wf in workflows]
                print(f"Fetched workflow IDs: {self.valid_workflow_ids}")
            else:
                print(f"Failed to fetch workflow_defs, status: {resp.status_code}")
                self.valid_workflow_ids = [1]
        except Exception as e:
            print(f"Exception fetching workflow_defs: {e}")
            self.valid_workflow_ids = [1]

    def auth_header(self):
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    @task
    def start_workflow(self):
        if not self.valid_workflow_ids or not self.token:
            return

        workflow_id = random.choice(self.valid_workflow_ids)
        fail_step = random.random() < 0.2
        step_action = "fail" if fail_step else "noop"
        step_id = random_string()

        dsl_payload = {
            "nodes": {
                step_id: {
                    "id": step_id,
                    "type": "task",
                    "action": step_action,
                    "should_fail": fail_step,
                    "incoming": []
                }
            },
            "edges": []
        }

        payload = {
            "workflow_id": workflow_id,
            "dsl": dsl_payload,
            "created_by": "locust",
            "version": "1.0"
        }

        try:
            resp = self.client.post("/flow/run/start", json=payload, headers=self.auth_header())
            if resp.status_code in [200, 201]:
                run_id = resp.json().get("run_id")
                if run_id:
                    self.run_ids.append(run_id)
                    print(f"[SUCCESS] Run started: {run_id} | Step fail={fail_step}")
            else:
                print(f"[FAIL] Status {resp.status_code} | {resp.text}")

        except Exception as e:
            print(f"[EXCEPTION] {e}")

    @task
    def check_run_status(self):
        if not self.run_ids:
            return  

        run_id = random.choice(self.run_ids)

        try:
            resp = self.client.get(f"/flow/run/{run_id}", headers=self.auth_header())
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status")
                steps = data.get("steps", [])
                print(f"[STATUS] Run {run_id} status: {status} | Steps: {len(steps)}")
            else:
                print(f"[STATUS FAIL] Status {resp.status_code} | {resp.text}")
        except Exception as e:
            print(f"[STATUS EXCEPTION] {e}")
