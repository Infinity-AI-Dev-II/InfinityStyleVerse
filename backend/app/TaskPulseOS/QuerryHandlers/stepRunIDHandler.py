

from backend.app.TaskPulseOS.Quarries.stepRunIDQuery import stepRunIDQuery
from backend.app.database import get_db_session
from backend.app.models.run_steps import RunStep


def stepRunIDHandler(command: stepRunIDQuery):
    try:
        with get_db_session() as session:
            results = session.query(RunStep).filter(
                RunStep.run_id == stepRunIDQuery.runID
            ).all()
            if results is None:
                return {"Status":"Success","Message":"Data is null"}
            else:
                return {"Status":"Success","Message":results}
    except Exception as e:
        return {"Status":"Error","Message":str(e)}

