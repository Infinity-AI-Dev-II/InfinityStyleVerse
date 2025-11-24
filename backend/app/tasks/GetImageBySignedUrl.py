import requests

# from backend import celery_app
from backend.celery_app import celery_app
#a celery task to get the image file (from AWS S3) using the signed URL.
#calles to get the img file using link
@celery_app.task(
    bind=True,
    name="backend.app.tasks.GetImageBySignedUrl.load_image_from_signed_url",
    autoretry_for=(Exception,),
    retry_backoff=True,
    max_retries=2,
    time_limit=30,
    soft_time_limit=60,
    queue="image_queue",
)
def load_image_from_signed_url(self, signed_url: str):
    r = requests.get(signed_url, timeout=30)
    r.raise_for_status()
    return r.content

