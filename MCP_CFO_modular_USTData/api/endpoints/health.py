"""
Health check API endpoints
"""
import asyncio
import time
from datetime import datetime
from fastapi import APIRouter, HTTPException

# These will be injected by main.py
db_manager = None
schema_manager = None

router = APIRouter()

@router.get("/health")
async def health_check():
    """Enhanced health check with chat session info"""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {},
        "schema_cache": {},
        "chat_session": {
            "session_id": "default-session-1234567890",
            "ready": True
        }
    }
    
    if not db_manager or not schema_manager:
        health_status["status"] = "degraded"
        health_status["services"]["initialization"] = "Services not properly injected"
        return health_status
    
    try:
        # Test SQL Database
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: db_manager.execute_sql_query("SELECT 1"))
        health_status["services"]["sql_database"] = "connected"
    except Exception as e:
        health_status["services"]["sql_database"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    try:
        # Test KQL Database
        test_result = await db_manager.test_kql_connection()
        health_status["services"]["kql_database"] = "connected" if test_result else "error"
        if not test_result:
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["kql_database"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Schema cache status
    if schema_manager.cached_tables_info is not None:
        cache_age = time.time() - (schema_manager.schema_cache_timestamp or 0)
        health_status["schema_cache"] = {
            "status": "loaded",
            "table_count": len(schema_manager.cached_tables_info),
            "cache_age_seconds": int(cache_age),
            "is_fresh": cache_age < schema_manager.schema_cache_duration
        }
    else:
        health_status["schema_cache"] = {
            "status": "not_loaded",
            "message": "Schema not cached yet"
        }
    
    # Quick chat history count
    try:
        count_query = f"""
        ChatHistory_CFO
        | where SessionID == 'default-session-1234567890'
        | where Question != 'tables_info' and Question != 'schema_info'
        | count
        """
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: db_manager.kusto_client.execute(db_manager.kusto_database, count_query)
        )
        
        if result.primary_results and len(result.primary_results[0]) > 0:
            chat_count = result.primary_results[0][0]["Count"]
            health_status["chat_session"]["stored_conversations"] = chat_count
    except:
        health_status["chat_session"]["stored_conversations"] = "unknown"
    
    health_status["features"] = [
        "Natural language processing", 
        "SQL analytics", 
        "Business insights", 
        "Smart visualization", 
        "KQL storage",
        "In-memory schema caching",
        "UI chat management"
    ]
    
    if health_status["status"] == "degraded":
        raise HTTPException(status_code=503, detail=health_status)
        
    return health_status