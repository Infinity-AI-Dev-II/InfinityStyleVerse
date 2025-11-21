from ..database import db

class BodyProfile(db.Model):
    __tablename__ = 'body_profiles'
    
    profile_id = db.Column(db.Integer, primary_key=True)
    body_type = db.Column(db.String(50), nullable=True)
    proportions_json = db.Column(db.JSON, nullable=True)
    posture_json = db.Column(db.JSON, nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    updated_at = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now())

    landmarks = db.relationship('Landmark', backref='body_profile', lazy=True)
    
    