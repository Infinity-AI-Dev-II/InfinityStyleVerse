# backend/tasks/sample.py
from backend.celery_app import celery_app
import time
from backend import app as app_pkg  # import to ensure app context if needed

@celery_app.task(bind=True)
def add(self, x, y):
    return x + y

@celery_app.task(bind=True)
def long_task(self, n=5):
    for i in range(n):
        self.update_state(state='PROGRESS', meta={'current': i+1, 'total': n})
        time.sleep(1)
    return {"result": "done"}
