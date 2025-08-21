"""
Chat management API endpoints
"""
import asyncio
import json
import traceback
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from utils.session_manager import SessionManager

# These will be injected by main.py
kql_storage = None
db_manager = None

router = APIRouter()

@router.get("/messages")
async def get_chat_messages(
    session: Optional[str] = Query(None, description="Session ID"),
    limit: Optional[int] = Query(10, description="Number of recent conversations to return")
):
    """Get chat messages for specified session with session validation"""
    
    if not kql_storage or not db_manager:
        raise HTTPException(status_code=500, detail="Required services not initialized")
    
    session_id = SessionManager.get_session_id_from_request(session)
    
    try:
        # First check if this session exists
        check_query = f"""
        ChatHistory_CFO
        | where SessionID == '{session_id}'
        | count
        """
        
        check_result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: db_manager.kusto_client.execute(db_manager.kusto_database, check_query)
        )
        
        session_exists = check_result.primary_results[0][0]["Count"] > 0 if check_result.primary_results[0] else False
        
        history_query = f"""
        ChatHistory_CFO
        | where SessionID == '{session_id}'
        | where Question != 'tables_info' and Question != 'schema_info'
        | order by Timestamp desc
        | take {limit * 2}
        | order by Timestamp asc
        | project Timestamp, Question, Response
        """
         
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: db_manager.kusto_client.execute(db_manager.kusto_database, history_query)
        )
        
        messages = []
        for row in result.primary_results[0]:
            try:
                response_data = json.loads(row["Response"])
                
                messages.append({
                    "id": f"user_{len(messages)}",
                    "type": "user",
                    "content": row["Question"],
                    "timestamp": row["Timestamp"]
                })
                
                messages.append({
                    "id": f"assistant_{len(messages)}",
                    "type": "assistant", 
                    "content": response_data.get("analysis", "No analysis available"),
                    "sql": response_data.get("generated_sql"),
                    "result_count": response_data.get("result_count", 0),
                    "sample_data": response_data.get("sample_data", []),
                    "visualization": response_data.get("visualization"),
                    "timestamp": row["Timestamp"]
                })
                
            except json.JSONDecodeError:
                continue
            
        return {
            "status": "success",
            "session_id": session_id,
            "session_exists": session_exists,
            "messages": messages,
            "message_count": len(messages),
            "total_pairs": len(messages) // 2
        }
        
    except Exception as e:
        return {
            "status": "error",
            "session_id": session_id,
            "session_exists": False,
            "messages": [],
            "message_count": 0,
            "total_pairs": 0,
            "error": str(e)
        }

@router.post("/clear")
async def clear_chat_and_start_new_session(
    session: Optional[str] = Query(None, description="Current Session ID"),
    create_new: Optional[bool] = Query(True, description="Create new session after clear")
):
    """Clear current session and optionally start a new one"""
    
    current_session_id = SessionManager.get_session_id_from_request(session)
    
    try:
        if create_new:
            # Generate a new session ID
            new_session_id = SessionManager.generate_new_session_id()
            
            return {
                "status": "success",
                "message": "Chat cleared and new session started",
                "old_session_id": current_session_id,
                "new_session_id": new_session_id,
                "timestamp": datetime.now().isoformat(),
                "action": "new_session_created"
            }
        else:
            return {
                "status": "success", 
                "message": "Chat cleared successfully",
                "session_id": current_session_id,
                "timestamp": datetime.now().isoformat(),
                "action": "session_cleared"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to clear chat")

@router.get("/sessions")
async def get_chat_sessions(
    date: Optional[str] = Query(None, description="Date in YYYYMMDD format, or 'all' for all sessions"),
    limit: Optional[int] = Query(50, description="Maximum number of sessions to return")
):
    """Get chat sessions with dynamic naming and full history support"""
    
    if not db_manager:
        raise HTTPException(status_code=500, detail="Database manager not initialized")
    
    try:
        if not date:
            date = "all"
            
        if date == "all":
            sessions_query = f"""
            ChatHistory_CFO
            | where SessionID startswith 'powerbi_'
            | where Question != 'tables_info' and Question != 'schema_info'
            | summarize 
                MessageCount = count(),
                FirstMessage = min(Timestamp),
                LastMessage = max(Timestamp),
                FirstQuestion = take_any(Question),
                LastQuestion = arg_max(Timestamp, Question)
            by SessionID
            | order by LastMessage desc
            | take {limit}
            """
        else:
            sessions_query = f"""
            ChatHistory_CFO
            | where SessionID contains 'powerbi_{date}'
            | where Question != 'tables_info' and Question != 'schema_info'
            | summarize 
                MessageCount = count(),
                FirstMessage = min(Timestamp),
                LastMessage = max(Timestamp),
                FirstQuestion = take_any(Question),
                LastQuestion = arg_max(Timestamp, Question)
            by SessionID
            | order by LastMessage desc
            | take {limit}
            """
        
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: db_manager.kusto_client.execute(db_manager.kusto_database, sessions_query)
        )
        
        raw_results = result.primary_results[0] if result.primary_results else []
        
        sessions = []
        for i, row in enumerate(raw_results):
            try:
                session_id = row["SessionID"]
                message_count = row["MessageCount"]
                first_message = row["FirstMessage"]
                last_message = row["LastMessage"]
                first_question = row["FirstQuestion"]
                last_question = row.get("LastQuestion", first_question) if hasattr(row, 'get') else row["LastQuestion"] if "LastQuestion" in row else first_question
                
                # Use last question for better identification, fallback to first question
                display_question = last_question or first_question or "Unknown"
                
                # Clean and truncate the question for display
                display_question = str(display_question).strip()
                if len(display_question) > 45:
                    display_question = display_question[:45] + "..."
                
                # Extract date from session ID for grouping
                session_parts = session_id.split('_')
                session_date = "Unknown"
                
                if len(session_parts) >= 2:
                    date_part = session_parts[1]
                    if len(date_part) == 8:  # YYYYMMDD format
                        try:
                            parsed_date = datetime.strptime(date_part, "%Y%m%d")
                            session_date = parsed_date.strftime("%b %d, %Y")
                        except:
                            session_date = date_part
                
                session_info = {
                    "session_id": session_id,
                    "display_name": display_question,
                    "message_count": message_count,
                    "first_message": first_message,
                    "last_message": last_message,
                    "first_question": first_question,
                    "last_question": last_question,
                    "session_date": session_date,
                    "is_today": session_date == datetime.now().strftime("%b %d, %Y")
                }
                sessions.append(session_info)
                
            except Exception as e:
                continue
        
        return {
            "status": "success",
            "query_type": "all" if date == "all" else f"date_{date}",
            "sessions": sessions,
            "total_sessions": len(sessions)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "query_type": "all" if date == "all" else f"date_{date}",
            "sessions": [],
            "total_sessions": 0,
            "error": str(e)
        }

@router.get("/messages")
async def get_chat_messages(
    session: Optional[str] = Query(None, description="Session ID"),
    limit: Optional[int] = Query(10, description="Number of recent conversations to return")
):
    """Get chat messages for specified session with session validation"""
    
    session_id = SessionManager.get_session_id_from_request(session)
    
    try:
        # First check if this session exists
        check_query = f"""
        ChatHistory_CFO
        | where SessionID == '{session_id}'
        | count
        """
        
        check_result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: db_manager.kusto_client.execute(db_manager.kusto_database, check_query)
        )
        
        session_exists = check_result.primary_results[0][0]["Count"] > 0 if check_result.primary_results[0] else False
        
        history_query = f"""
        ChatHistory_CFO
        | where SessionID == '{session_id}'
        | where Question != 'tables_info' and Question != 'schema_info'
        | order by Timestamp desc
        | take {limit * 2}
        | order by Timestamp asc
        | project Timestamp, Question, Response
        """
         
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: db_manager.kusto_client.execute(db_manager.kusto_database, history_query)
        )
        
        messages = []
        for row in result.primary_results[0]:
            try:
                response_data = json.loads(row["Response"])
                
                messages.append({
                    "id": f"user_{len(messages)}",
                    "type": "user",
                    "content": row["Question"],
                    "timestamp": row["Timestamp"]
                })
                
                messages.append({
                    "id": f"assistant_{len(messages)}",
                    "type": "assistant", 
                    "content": response_data.get("analysis", "No analysis available"),
                    "sql": response_data.get("generated_sql"),
                    "result_count": response_data.get("result_count", 0),
                    "sample_data": response_data.get("sample_data", []),
                    "visualization": response_data.get("visualization"),
                    "timestamp": row["Timestamp"]
                })
                
            except json.JSONDecodeError:
                continue
            
        return {
            "status": "success",
            "session_id": session_id,
            "session_exists": session_exists,
            "messages": messages,
            "message_count": len(messages),
            "total_pairs": len(messages) // 2
        }
        
    except Exception as e:
        return {
            "status": "error",
            "session_id": session_id,
            "session_exists": False,
            "messages": [],
            "message_count": 0,
            "total_pairs": 0,
            "error": str(e)
        }

@router.post("/clear")
async def clear_chat_and_start_new_session(
    session: Optional[str] = Query(None, description="Current Session ID"),
    create_new: Optional[bool] = Query(True, description="Create new session after clear")
):
    """Clear current session and optionally start a new one"""
    
    current_session_id = SessionManager.get_session_id_from_request(session)
    
    try:
        if create_new:
            # Generate a new session ID
            new_session_id = SessionManager.generate_new_session_id()
            
            return {
                "status": "success",
                "message": "Chat cleared and new session started",
                "old_session_id": current_session_id,
                "new_session_id": new_session_id,
                "timestamp": datetime.now().isoformat(),
                "action": "new_session_created"
            }
        else:
            return {
                "status": "success", 
                "message": "Chat cleared successfully",
                "session_id": current_session_id,
                "timestamp": datetime.now().isoformat(),
                "action": "session_cleared"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to clear chat")

@router.get("/sessions")
async def get_chat_sessions(
    date: Optional[str] = Query(None, description="Date in YYYYMMDD format, or 'all' for all sessions"),
    limit: Optional[int] = Query(50, description="Maximum number of sessions to return")
):
    """Get chat sessions with dynamic naming and full history support"""
    
    try:
        if not date:
            date = "all"
            
        if date == "all":
            sessions_query = f"""
            ChatHistory_CFO
            | where SessionID startswith 'powerbi_'
            | where Question != 'tables_info' and Question != 'schema_info'
            | summarize 
                MessageCount = count(),
                FirstMessage = min(Timestamp),
                LastMessage = max(Timestamp),
                FirstQuestion = take_any(Question),
                LastQuestion = arg_max(Timestamp, Question)
            by SessionID
            | order by LastMessage desc
            | take {limit}
            """
        else:
            sessions_query = f"""
            ChatHistory_CFO
            | where SessionID contains 'powerbi_{date}'
            | where Question != 'tables_info' and Question != 'schema_info'
            | summarize 
                MessageCount = count(),
                FirstMessage = min(Timestamp),
                LastMessage = max(Timestamp),
                FirstQuestion = take_any(Question),
                LastQuestion = arg_max(Timestamp, Question)
            by SessionID
            | order by LastMessage desc
            | take {limit}
            """
        
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: db_manager.kusto_client.execute(db_manager.kusto_database, sessions_query)
        )
        
        raw_results = result.primary_results[0] if result.primary_results else []
        
        sessions = []
        for i, row in enumerate(raw_results):
            try:
                session_id = row["SessionID"]
                message_count = row["MessageCount"]
                first_message = row["FirstMessage"]
                last_message = row["LastMessage"]
                first_question = row["FirstQuestion"]
                last_question = row.get("LastQuestion", first_question) if hasattr(row, 'get') else row["LastQuestion"] if "LastQuestion" in row else first_question
                
                # Use last question for better identification, fallback to first question
                display_question = last_question or first_question or "Unknown"
                
                # Clean and truncate the question for display
                display_question = str(display_question).strip()
                if len(display_question) > 45:
                    display_question = display_question[:45] + "..."
                
                # Extract date from session ID for grouping
                session_parts = session_id.split('_')
                session_date = "Unknown"
                
                if len(session_parts) >= 2:
                    date_part = session_parts[1]
                    if len(date_part) == 8:  # YYYYMMDD format
                        try:
                            parsed_date = datetime.strptime(date_part, "%Y%m%d")
                            session_date = parsed_date.strftime("%b %d, %Y")
                        except:
                            session_date = date_part
                
                session_info = {
                    "session_id": session_id,
                    "display_name": display_question,
                    "message_count": message_count,
                    "first_message": first_message,
                    "last_message": last_message,
                    "first_question": first_question,
                    "last_question": last_question,
                    "session_date": session_date,
                    "is_today": session_date == datetime.now().strftime("%b %d, %Y")
                }
                sessions.append(session_info)
                
            except Exception as e:
                continue
        
        return {
            "status": "success",
            "query_type": "all" if date == "all" else f"date_{date}",
            "sessions": sessions,
            "total_sessions": len(sessions)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "query_type": "all" if date == "all" else f"date_{date}",
            "sessions": [],
            "total_sessions": 0,
            "error": str(e)
        }