"""
Logging configuration for Intelligent Fabric Analytics
"""
import structlog

def setup_logging():
    """Setup structured logging"""
    structlog.configure(
        processors=[structlog.processors.JSONRenderer()],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    
    return structlog.get_logger()