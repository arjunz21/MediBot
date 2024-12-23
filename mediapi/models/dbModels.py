from datetime import datetime
from mediapi.models import base
from mediapi.models import Boolean, Column, Integer, String, TIMESTAMP, DateTime

class User(base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True)
    dated = Column(TIMESTAMP)