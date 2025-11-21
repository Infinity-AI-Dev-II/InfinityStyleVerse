from flask import Blueprint, request, jsonify
import uuid
import random
from flask_jwt_extended import jwt_required, get_jwt_identity

bodyMorph_bp = Blueprint('bodyMorph_bp', __name__, url_prefix="/stylesense")

# Maximum file size (for frontend validation reference)
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Mock BodyMorph analysis endpoint
@bodyMorph_bp.route("/body_profile", methods=["POST"])
@jwt_required() 
def body_profile():
    """
    Accepts JSON payload with image_uri, hints, and camera info.
    Returns mocked body analysis results.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400
    file = request.files['image']
    
    data = request.get_json()
    if not data or "hints" not in data or "gender" not in data["hints"]:
        return jsonify({"error": "Invalid payload"}), 400
    
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
    return jsonify(response)


