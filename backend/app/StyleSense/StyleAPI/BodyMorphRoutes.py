import io
import json
from flask import Blueprint, Config, request, jsonify,abort
import uuid
import random
from flask_jwt_extended import jwt_required, get_jwt_identity

import time
import hmac
import hashlib
from backend.app.AWS_configuration import AWSConfig
from backend.app.services.idempotency_service import compute_request_hash, read_idempotency, write_idempotency

import boto3

from backend.app.tasks.GetImageBySignedUrl import load_image_from_signed_url
# from botocore.exceptions import NoCredentialsError, PartialCredentialsError
# from botocore.client import Config
import uuid
import os
# from flask import current_app
from backend.app.requestSizeValidator import validate_request_size
from backend.app.services.redisKeyGenerate import generate_image_cache_key
import redis
import hashlib
bodyMorph_bp = Blueprint('bodyMorph_bp', __name__, url_prefix="/stylesense")

# Maximum file size (for frontend validation reference)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Initialize Boto3 S3 client
s3_client = AWSConfig.get_s3_client()
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')

def generate_presigned_url(object_key, expiration=3600):
    """Generate a pre-signed URL to upload a file to S3."""
    try:
        return s3_client.generate_presigned_url(
            'put_object',
            Params={'Bucket': S3_BUCKET_NAME, 'Key': object_key},
            ExpiresIn=expiration
        )
    # except (NoCredentialsError, PartialCredentialsError) as e:
    #         return f"Credentials error: {str(e)}"
    except Exception as e:
        print(f"Error generating pre-signed URL: {e}")
        return None
    
# create a presigned url to store the image in the S3
@bodyMorph_bp.route('/generate-presigned-url', methods=['GET'])
# @jwt_required()
@validate_request_size(request,max_json_kb=500)
def generate_presigned_url_api():
    """Generate a presigned URL for S3 file upload."""
    try:
        # Get file extension from query parameters
        file_extension = request.args.get('file_extension')
        if not file_extension:
            return jsonify({"message": "file_extension is required"}), 400

        # Ensure file extension starts with a dot
        file_extension = f".{file_extension}"

        # Get optional folder path if provided
        folder_path = request.args.get('folder_path', '').strip()
        
        # Generate a unique filename using UUID
        unique_id = str(uuid.uuid4())
        new_file_name = f"{unique_id}{file_extension}"

        # Set the object key with folder path if specified
        object_key = f"{folder_path}/{new_file_name}" if folder_path else new_file_name

        # Generate the pre-signed URL
        presigned_url = generate_presigned_url(object_key)
        if not presigned_url:
            return jsonify({"message": "Failed to generate presigned URL"}), 500

        return jsonify({
            "presigned_url": presigned_url,
            "object_key": object_key
        }), 200

    except Exception as e:
        return jsonify({"message": f"Error generating presigned URL: {str(e)}"}), 500



@bodyMorph_bp.route("/body_profile", methods=["POST"])
# @jwt_required() 
# @validate_request_size(request,max_json_kb=500)
def body_profile():
    """
    Accepts JSON payload with image_uri, hints, and camera info.
    Returns mocked body analysis results.
    """
    #FIXME: TEST ONLY
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in request'}), 400

    file_like = request.files['image']
    #redis client
    # redis_client = current_app.config["REDIS_CLIENT"]
    
    #FIXME: TEST ONLY
    redis_client = redis.Redis(host='redis', port=6379, db=0)
    
    #Idempotency-Key checking
    idem_key = request.headers.get("Idempotency-Key")
    if not idem_key:
        return jsonify({"Error:": "Idempotency-Key is empty"})
    # if 'image' not in request.files:
    #     return jsonify({'error': 'No image part in the request'}), 400
    if idem_key:
        status = read_idempotency(idem_key)
        if status:
            try:
                response = json.loads(status.response_json)
            except:
                response = {"Error" : "Error occured"}
            return response    
    #FIXME: REMOVE THIS LATER
    #chack the information availability
    # data = request.get_json()
    # if not data or "hints" not in data or "image_uri" not in data or "gender" not in data["hints"]:
    #     return jsonify({"error": "Invalid payload"}), 400
    
    #get the image file using the signed url
    #FIXME: FOR NOW
    # imag_file = load_image_from_signed_url(data['image_uri']) 
    # file_like = io.BytesIO(imag_file)
    
    #FIXME: TEST REQUEST BODY
    # body = request.get_json()
    body = { 
             "image_uri": "s3://bucket/uid/front.jpg", 
             "hints": {"gender":"F","height_cm":168}, 
             "camera": {"fov_deg":60, "distance_m":2.0} 
           }
    hints = body.get("hints",{})#hints from the request
    imgBytes = file_like.read()#img bytes from the cloud file
    #cache key generation
    key = generate_image_cache_key(image_bytes=imgBytes,hints=hints)
    
    #chack cache
    cacheFound = redis_client.get(key)
    if cacheFound:
        return jsonify(json.loads(cacheFound))
    
    
    #TODO: ML SERVICES
    #TODO: mock the services
    
    #test output
    response = { 
                "body_type": "hourglass", 
                "confidence": 0.82, 
                "landmarks": { "shoulder_L":[1,2], "shoulder_R":[1,2], "hip_L":[1,2], 
                "hip_R":[1,2], "waist":[1,2], "knee_L":[1,2], "ankle_L":[1,2]}, 
                "proportions": { "shoulder_width_px": 412, "hip_width_px": 405, 
                "waist_width_px": 315, "torso_len_px": 670, "leg_len_px": 880 }, 
                "normalized": { "shoulder_to_hip_ratio": 1.02, "waist_to_hip_ratio": 0.78, 
                "torso_to_leg_ratio": 0.76 }, 
                "posture": { "tilt":"neutral", "slouch":"low", "stance":"closed" }, 
                "occlusion": { "percent": 0.11, "regions":["lower_arm_R"] }, 
                "quality_flags": ["single_subject","front_view_detected","lighting_ok"] 
            }
    request_hash_str = json.dumps(body, sort_keys=True)
    request_hash = hashlib.sha256(request_hash_str.encode()).hexdigest()
    #create the ideopitency row
    write_idempotency(idem_key, compute_request_hash(request_hash), response)
    #cache the output with the key
    print("Server starting...", flush=True)
    redis_client.set(key, json.dumps(response), ex=3600)
    return jsonify(response)


