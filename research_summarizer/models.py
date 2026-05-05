from extensions import db
from flask_login import UserMixin
from datetime import datetime
import json

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    datasets = db.relationship('Dataset', backref='owner', lazy=True)

class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    extracted_text = db.Column(db.Text, nullable=False)
    summary_data = db.Column(db.Text, nullable=True) # JSON string of summary result
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_summary(self, summary_dict):
        self.summary_data = json.dumps(summary_dict)
        
    def get_summary(self):
        return json.loads(self.summary_data) if self.summary_data else None
