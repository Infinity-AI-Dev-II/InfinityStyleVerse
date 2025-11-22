import os
import boto3
from botocore.client import Config
from threading import Lock
#configures AWS S3 client to use globally + singalton pattern
class AWSConfig:
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
    S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
    
    #instance obj
    _instance = None
    #lock obj (mutual exclusion (mutex) lock object): only one thread at a time can execute a critical section of code 
    _lock = Lock()
    
    @classmethod
    def get_s3_client(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = boto3.client(
                                    's3',
                                    aws_access_key_id=cls.AWS_ACCESS_KEY_ID,
                                    aws_secret_access_key=cls.AWS_SECRET_ACCESS_KEY,
                                    region_name=cls.AWS_REGION,
                                    config=Config(signature_version='s3v4')
                                    )
        return cls._instance