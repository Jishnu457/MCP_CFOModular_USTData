"""
Admin API endpoints
"""
import asyncio
from datetime import datetime
from fastapi import APIRouter, HTTPException
import structlog

# These will be injected by main.py
db_manager = None
kql_storage = None
schema_manager = None

logger = structlog.get_logger()
router = APIRouter()

@router.post("/schema/refresh")
async def refresh_schema_cache():
    """Manually refresh the schema cache"""
    if not schema_manager:
        raise HTTPException(status_code=500, detail="Schema manager not initialized")
        
    try:
        logger.info("Manual schema refresh requested")
        
        # Clear current cache
        schema_manager.refresh_cache()
        
        # Fetch fresh schema
        tables_info = await schema_manager.get_cached_tables_info()
        
        return {
            "status": "success",
            "message": "Schema cache refreshed successfully",
            "table_count": len(tables_info),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Schema refresh failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Schema refresh failed: {str(e)}")

@router.delete("/cache/clear")
async def admin_clear_kql_cache():
    """ADMIN ONLY: Clear the entire KQL ChatHistory_CFO table"""
    if not db_manager or not kql_storage:
        raise HTTPException(status_code=500, detail="Required services not initialized")
        
    try:
        clear_query = ".drop table ChatHistory_CFO"
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: db_manager.kusto_client.execute(db_manager.kusto_database, clear_query)
        )
        await kql_storage.initialize_kql_table()
        
        logger.warning("ADMIN: KQL cache cleared completely")
        
        return {
            "status": "success",
            "message": "KQL cache cleared completely - ALL conversation history deleted",
            "timestamp": datetime.now().isoformat(),
            "warning": "This action cannot be undone"
        }
    except Exception as e:
        logger.error("Admin KQL cache clear failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to clear KQL cache: {str(e)}")

@router.get("/debug/schema-order")
async def debug_schema_order(question: str = "Create a P&L report for 2025"):
    """Debug schema ordering for troubleshooting"""
    if not schema_manager:
        raise HTTPException(status_code=500, detail="Schema manager not initialized")
        
    try:
       
        from services.prompt_manager import PromptManager
        from services.ai_services import AIServiceManager
        
        ai_services = AIServiceManager()
        prompt_manager = PromptManager(ai_services)
        
        tables_info = await schema_manager.get_cached_tables_info()
        relevant_tables = prompt_manager.filter_schema_for_question(question, tables_info)
        
        result = []
        for i, table in enumerate(relevant_tables[:5]):
            table_name = table.get('table', '')
            columns = table.get('columns', [])
            
            # Check for financial columns
            financial_cols = []
            for col in columns:
                col_text = col.lower()
                if any(term in col_text for term in ['revenue', 'profit', 'expense', 'income']):
                    financial_cols.append(col.split()[0])  # Just column name
            
            result.append({
                "order": i + 1,
                "table": table_name,
                "total_columns": len(columns),
                "financial_columns": financial_cols,
                "is_balance_sheet": 'balance' in table_name.lower()
            })
        
        return {
            "question": question,
            "tables_in_order": result,
            "note": "AI picks first table with financial columns"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/schema/refresh")
async def refresh_schema_cache():
    """Manually refresh the schema cache"""
    try:
        logger.info("Manual schema refresh requested")
        
        # Clear current cache
        schema_manager.refresh_cache()
        
        # Fetch fresh schema
        tables_info = await schema_manager.get_cached_tables_info()
        
        return {
            "status": "success",
            "message": "Schema cache refreshed successfully",
            "table_count": len(tables_info),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Schema refresh failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Schema refresh failed: {str(e)}")

@router.delete("/cache/clear")
async def admin_clear_kql_cache():
    """ADMIN ONLY: Clear the entire KQL ChatHistory_CFO table"""
    try:
        clear_query = ".drop table ChatHistory_CFO"
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: db_manager.kusto_client.execute(db_manager.kusto_database, clear_query)
        )
        await kql_storage.initialize_kql_table()
        
        logger.warning("ADMIN: KQL cache cleared completely")
        
        return {
            "status": "success",
            "message": "KQL cache cleared completely - ALL conversation history deleted",
            "timestamp": datetime.now().isoformat(),
            "warning": "This action cannot be undone"
        }
    except Exception as e:
        logger.error("Admin KQL cache clear failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to clear KQL cache: {str(e)}")

@router.get("/debug/schema-order")
async def debug_schema_order(question: str = "Create a P&L report for 2025"):
    """Debug schema ordering for troubleshooting"""
    try:
        from services.prompt_manager import prompt_manager
        
        tables_info = await schema_manager.get_cached_tables_info()
        relevant_tables = prompt_manager.filter_schema_for_question(question, tables_info)
        
        result = []
        for i, table in enumerate(relevant_tables[:5]):
            table_name = table.get('table', '')
            columns = table.get('columns', [])
            
            # Check for financial columns
            financial_cols = []
            for col in columns:
                col_text = col.lower()
                if any(term in col_text for term in ['revenue', 'profit', 'expense', 'income']):
                    financial_cols.append(col.split()[0])  # Just column name
            
            result.append({
                "order": i + 1,
                "table": table_name,
                "total_columns": len(columns),
                "financial_columns": financial_cols,
                "is_balance_sheet": 'balance' in table_name.lower()
            })
        
        return {
            "question": question,
            "tables_in_order": result,
            "note": "AI picks first table with financial columns"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))