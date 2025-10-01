from locust import HttpUser, task, between
import random
import string
import requests
import json

API_HOST = "http://api:8000"

def random_string(length=6):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

class WorkflowUser(HttpUser):
    wait_time = between(1, 3)
    host = API_HOST
    valid_workflow_ids = []

    def on_start(self):
        try:
            response = requests.get(f"{API_HOST}/flow/workflow_defs")
            if response.status_code == 200:
                workflows = response.json()
                self.valid_workflow_ids = [wf["id"] for wf in workflows]
                print(f"Fetched workflow IDs: {self.valid_workflow_ids}")
            else:
                print(f"Failed to fetch workflow_defs, status: {response.status_code}")
                self.valid_workflow_ids = [1]
        except Exception as e:
            print(f"Exception fetching workflow_defs: {e}")
            self.valid_workflow_ids = [1]

    @task
    def start_workflow(self):
        if not self.valid_workflow_ids:
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
            response = self.client.post("/flow/run/start", json=payload)
            if response.status_code == 200:
                run_id = response.json().get("run_id")
                print(f"[SUCCESS] Run started: {run_id} | Step fail={fail_step}")
            else:
                print(f"[FAIL] Status {response.status_code} | {response.text}")
        except Exception as e:
            print(f"[EXCEPTION] {e}")

    @task
    def check_run_status(self):
        run_id = random.randint(1, 50)
        try:
            resp =self.client.get(f"/flow/flow/run/{run_id}")
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status")
                steps = data.get("steps", [])
                print(f"[STATUS] Run {run_id} status: {status} | Steps: {len(steps)}")
            else:
                print(f"[STATUS FAIL] Status {resp.status_code} | {resp.text}")
        except Exception as e:
            print(f"[STATUS EXCEPTION] {e}")
