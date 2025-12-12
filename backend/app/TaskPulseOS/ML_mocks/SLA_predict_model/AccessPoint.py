#inner endpoint which accesses the SLA predict model (in both mock and real versions)
from flask import Blueprint, request
from backend.app.Decorators.routesToInternal import limmit_route_internal
from backend.app.TaskPulseOS.ML_mocks.SLA_predict_model.model import predict_sla
access_point_bp = Blueprint('access_point_bp', __name__)

@access_point_bp.route('/predict/sla_risk', methods=['POST'])
@limmit_route_internal
def predict_sla_risk():
    runSteps = request.json.get('run_steps', [])
    workflowDef = request.json.get('workflow_def', {})
    result = predict_sla(
    run_steps=runSteps,
    workflow_def=workflowDef
    )   
    return result, 200