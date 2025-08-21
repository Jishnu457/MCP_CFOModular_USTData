"""
Pydantic request models
"""
import re
from typing import List, Optional
from pydantic import BaseModel, validator

class IntelligentRequest(BaseModel):
    question: str
    enable_ai_insights: Optional[bool] = True
    enable_email_notification: Optional[bool] = False
    email_recipients: Optional[List[str]] = []

    @validator("question")
    def validate_question(cls, value):
        if not value.strip():
            raise ValueError("Question cannot be empty")
        if len(value) < 3:
            raise ValueError("Question is too short; please provide more details")
        return value.strip()

class ReportRequest(BaseModel):
    data_query: str
    report_type: Optional[str] = "executive"  # executive, detailed, summary
    report_format: Optional[str] = "pdf"  # pdf, excel, both
    email_recipients: List[str]
    subject_hint: Optional[str] = None
    include_ai_analysis: Optional[bool] = True

    @validator("email_recipients")
    def validate_emails(cls, value):
        if not value:
            raise ValueError("At least one email recipient is required")
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        for email in value:
            if not email_pattern.match(email):
                raise ValueError(f"Invalid email address: {email}")
        return value