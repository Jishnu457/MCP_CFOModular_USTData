"""
Pydantic response models
"""
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel

class AnalyticsResponse(BaseModel):
    """Standard analytics response model"""
    question: str
    generated_sql: Optional[str] = None
    analysis: str
    result_count: Optional[int] = 0
    sample_data: Optional[List[Dict[str, Any]]] = []
    timestamp: str
    session_id: str
    ai_insights_enabled: bool
    response_type: Optional[str] = "analytics"
    
    # Enhanced features
    enhanced_analysis: Optional[str] = None
    ai_insights: Optional[str] = None
    conversation_context_used: Optional[bool] = False
    
    # Visualization
    has_visualization: Optional[bool] = False
    visualization: Optional[Dict[str, Any]] = None
    chart_explanation: Optional[str] = None
    chart_type: Optional[str] = None
    
    # Error handling
    error: Optional[str] = None
    suggestion: Optional[str] = None

class ConversationalResponse(BaseModel):
    """Response for conversational/non-data questions"""
    question: str
    response_type: str = "conversational"
    analysis: str
    timestamp: str
    session_id: str
    ai_insights_enabled: bool
    ai_insights: Optional[str] = None
    conversation_history: Optional[List[Dict]] = []

class ErrorResponse(BaseModel):
    """Error response model"""
    question: str
    error: str
    analysis: str
    suggestion: str
    session_id: str
    response_type: str = "error"
    timestamp: str
    ai_insights_enabled: bool

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str  # healthy, degraded, unhealthy
    timestamp: str
    services: Dict[str, str]
    schema_cache: Dict[str, Any]
    chat_session: Dict[str, Any]
    features: List[str]

class ChatMessage(BaseModel):
    """Individual chat message model"""
    id: str
    type: str  # user or assistant
    content: str
    timestamp: Union[str, datetime]
    
    # Assistant-specific fields
    sql: Optional[str] = None
    result_count: Optional[int] = None
    sample_data: Optional[List[Dict]] = []
    visualization: Optional[Dict] = None

class ChatHistoryResponse(BaseModel):
    """Chat history response model"""
    status: str
    session_id: str
    session_exists: bool
    messages: List[ChatMessage]
    message_count: int
    total_pairs: int
    error: Optional[str] = None

class SessionInfo(BaseModel):
    """Session information model"""
    session_id: str
    display_name: str
    message_count: int
    first_message: Union[str, datetime]
    last_message: Union[str, datetime]
    first_question: str
    last_question: str
    session_date: str
    is_today: bool

class SessionsResponse(BaseModel):
    """Sessions list response model"""
    status: str
    query_type: str
    sessions: List[SessionInfo]
    total_sessions: int
    error: Optional[str] = None

class WorkflowResponse(BaseModel):
    """Workflow response model"""
    status: str
    workflow_id: Optional[str] = None
    message: str
    timestamp: str
    expected_filename: Optional[str] = None

class CapabilitiesResponse(BaseModel):
    """System capabilities response model"""
    capabilities: str
    example_questions: List[str]
    calculation_features: List[str]
    intelligence_features: List[str]
    visualization_features: List[str]
    supported_analysis: Optional[List[str]] = []

class ClearChatResponse(BaseModel):
    """Clear chat response model"""
    status: str
    message: str
    old_session_id: Optional[str] = None
    new_session_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: str
    action: str

class SchemaRefreshResponse(BaseModel):
    """Schema refresh response model"""
    status: str
    message: str
    table_count: int
    timestamp: str

class CacheClearResponse(BaseModel):
    """Cache clear response model"""
    status: str
    message: str
    timestamp: str
    warning: Optional[str] = None

class DebugResponse(BaseModel):
    """Debug response model"""
    question: str
    tables_in_order: List[Dict[str, Any]]
    note: Optional[str] = None
    issue: Optional[str] = None

class FeaturesEnabledResponse(BaseModel):
    """Features enabled response model"""
    ai_insights: bool
    email_notification: bool
    ai_foundry_available: bool
    graph_api_available: bool
    chat_context: bool

# Union type for all possible responses
AnalyticsResponseUnion = Union[
    AnalyticsResponse,
    ConversationalResponse,
    ErrorResponse
]