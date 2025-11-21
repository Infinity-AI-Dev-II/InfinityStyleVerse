from ..database import db

class Landmark(db.Model):
    __tablename__ = 'landmarks'
    
    id = db.Column(db.Integer, primary_key=True)
    profile_id = db.Column(db.Integer, db.ForeignKey('body_profiles.profile_id'), nullable=False)
    request_id = db.Column(db.Integer, nullable=True)
    points_json = db.Column(db.JSON, nullable=True)
    occlusion_json = db.Column(db.JSON, nullable=True)
    quality_flags = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    
    
    