from flask import Blueprint, request, jsonify, g
from flask_jwt_extended import jwt_required
from tasks.anomaly_detector_functions import detect_anomalies

flowgate_bp = Blueprint("flowGate", __name__, url_prefix="/flowGate")
#endpoint which gets the details about the anomaly detection
@flowgate_bp.route("/detectAnomalies",methods=["POST"])
@jwt_required(optional=True)
def detectingProcess():
    try:
       anomalies = detect_anomalies.delay()
       return jsonify(anomalies.get())
    except Exception as e:
        return jsonify({"error": "failed to detect anomalies", "detail": str(e)}), 500

