"""
KQL storage operations for conversation history
"""
import asyncio
import json
import time
import uuid
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional
import structlog
from azure.kusto.data.exceptions import KustoServiceError

from utils.helpers import Utils

logger = structlog.get_logger()

class KQLStorage:
    """Centralized KQL storage operations"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
    
    async def initialize_kql_table(self):
        """Create ChatHistory_CFO table if it doesn't exist"""
        create_table_query = """
        .create table ChatHistory_CFO (
            SessionID: string,
            Timestamp: datetime,
            ConversationID: string,
            Question: string,
            Response: string,
            Context: string
        )
        """
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.db_manager.kusto_client.execute(self.db_manager.kusto_database, create_table_query)
            )
            logger.info("KQL table ChatHistory_CFO created or verified")
        except KustoServiceError as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg or "entityalreadyexists" in error_msg:
                logger.info("KQL table ChatHistory_CFO already exists")
            else:
                logger.error("Failed to create KQL table", error=str(e))
                raise
        except Exception as e:
            logger.error("Unexpected error creating KQL table", error=str(e))
            raise
    
    async def get_last_query_response(self, session_id: str = None) -> Dict[str, Any]:
        """Get the most recent query response from KQL for context"""
        actual_session_id = session_id if session_id else "default-session-1234567890"
        
        last_response_query = f"""
        ChatHistory_CFO
        | where SessionID == '{actual_session_id}'
        | where Question != 'tables_info' and Question != 'schema_info'
        | order by Timestamp desc
        | take 1
        | project Question, Response, Context, Timestamp
        """
        
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.db_manager.kusto_client.execute(self.db_manager.kusto_database, last_response_query)
            )
            
            if result.primary_results and len(result.primary_results[0]) > 0:
                row = result.primary_results[0][0]
                
                # Parse the stored response
                response_data = json.loads(row["Response"])
                context_data = json.loads(row["Context"]) if row.get("Context") else {}
                
                return {
                    "previous_question": row["Question"],
                    "previous_response": response_data,
                    "previous_context": context_data,
                    "timestamp": row["Timestamp"],
                    "has_data": True
                }
            
            return {"has_data": False}
            
        except Exception as e:
            logger.error("Failed to retrieve last query response", error=str(e), session_id=actual_session_id)
            return {"has_data": False}
    
    async def get_recent_query_responses(self, session_id: str = None, limit: int = 3) -> List[Dict[str, Any]]:
        """Get the last N query responses from KQL for richer context"""
        actual_session_id = session_id if session_id else "default-session-1234567890"
        
        recent_responses_query = f"""
        ChatHistory_CFO
        | where SessionID == '{actual_session_id}'
        | where Question != 'tables_info' and Question != 'schema_info'
        | order by Timestamp desc
        | take {limit}
        | order by Timestamp asc
        | project Question, Response, Context, Timestamp
        """
        
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.db_manager.kusto_client.execute(self.db_manager.kusto_database, recent_responses_query)
            )
            
            responses = []
            for row in result.primary_results[0]:
                try:
                    response_data = json.loads(row["Response"])
                    context_data = json.loads(row["Context"]) if row.get("Context") else {}
                    
                    responses.append({
                        "question": row["Question"],
                        "response": response_data,
                        "context": context_data,
                        "timestamp": row["Timestamp"]
                    })
                    
                except json.JSONDecodeError:
                    continue
            
            logger.info("Retrieved recent responses for context", 
                       session_id=actual_session_id, 
                       count=len(responses))
            return responses
            
        except Exception as e:
            logger.error("Failed to retrieve recent responses", error=str(e), session_id=actual_session_id)
            return []

    async def store_in_kql(self, question: str, response: Dict, context: List[Dict], session_id: str = None):
        """Store query and response in KQL with base64 encoding"""
        # Skip storing schema-related queries
        if question.lower() in ['tables_info', 'schema_info'] or 'tables_info' in str(response):
            return
        
        # Use provided session ID or fall back to fixed session
        actual_session_id = session_id if session_id else "default-session-1234567890"
        
        conversation_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            # Better context extraction
            if response.get('sample_data'):
                extracted_context = Utils.extract_context_from_results(response['sample_data'])
            else:
                extracted_context = context if isinstance(context, dict) else {}
            
            # Safely serialize
            response_json = json.dumps(response, default=Utils.safe_json_serialize, ensure_ascii=False)
            context_json = json.dumps(extracted_context, default=Utils.safe_json_serialize, ensure_ascii=False)
        
            # Clean session ID
            clean_session_id = str(actual_session_id).strip()
            
            if clean_session_id.startswith('"') and clean_session_id.endswith('"'):
                clean_session_id = clean_session_id[1:-1]
            if clean_session_id.startswith("'") and clean_session_id.endswith("'"):
                clean_session_id = clean_session_id[1:-1]
                
            clean_session_id = clean_session_id.replace('"', '').replace("'", "")
            
            # Use base64 encoding to prevent JSON corruption
            clean_question = question.replace('\n', ' ').replace('\r', ' ').strip()
            
            # Encode JSON as base64 to avoid CSV parsing issues
            response_b64 = base64.b64encode(response_json.encode('utf-8')).decode('ascii')
            context_b64 = base64.b64encode(context_json.encode('utf-8')).decode('ascii')
            
            # Store base64 encoded data
            ingest_query = f'''.ingest inline into table ChatHistory_CFO <|
    {clean_session_id},datetime({timestamp}),{conversation_id},{clean_question},{response_b64},{context_b64}'''
            
            # Execute the ingest
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.db_manager.kusto_client.execute(self.db_manager.kusto_database, ingest_query)
            )
            
            # Verification
            verify_query = f"""
            ChatHistory_CFO
            | where SessionID has '{clean_session_id}'
            | where ConversationID == '{conversation_id}'
            | extend Decoded_Response = base64_decode_tostring(Response)
            | project SessionID, Question, Decoded_Response
            """
            
            await asyncio.sleep(1)
            
            verify_result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.db_manager.kusto_client.execute(self.db_manager.kusto_database, verify_query)
            )
            
            verify_records = verify_result.primary_results[0] if verify_result.primary_results else []
            
            logger.info("KQL storage successful with base64 encoding", 
                    session_id=clean_session_id,
                    conversation_id=conversation_id,
                    verification_count=len(verify_records))
            
        except Exception as e:
            logger.error("KQL storage failed", error=str(e))
            raise

    async def get_from_kql_cache(self, question: str, session_id: str = None) -> Optional[Dict]:
        """Retrieve cached response from KQL"""
        actual_session_id = session_id if session_id else "default-session-1234567890"
        normalized_question = Utils.normalize_question(question)
        
        cache_query = f"""
        ChatHistory_CFO
        | where SessionID == '{actual_session_id}'
        | where Question == '{normalized_question.replace("'", "''")}'
        | project Response
        | take 1
        """
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.db_manager.kusto_client.execute(self.db_manager.kusto_database, cache_query)
            )
            if result.primary_results and len(result.primary_results[0]) > 0:
                response = json.loads(result.primary_results[0][0]["Response"])
                response["session_id"] = actual_session_id
                logger.info("KQL cache hit", question=normalized_question, session_id=actual_session_id)
                return response
            return None
        except Exception as e:
            logger.error("KQL cache retrieval failed", error=str(e), session_id=actual_session_id)
            return None
    
    async def get_latest_responses(self, session_id: str = None) -> List[Dict]:
        """Retrieve latest 10 responses for UI"""
        actual_session_id = session_id if session_id else "default-session-1234567890"
        
        history_query = f"""
        ChatHistory_CFO
        | where SessionID == '{actual_session_id}'
        | order by Timestamp desc
        | take 10
        | project Timestamp, Question, Response
        """
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.db_manager.kusto_client.execute(self.db_manager.kusto_database, history_query)
            )
            responses = []
            for row in result.primary_results[0]:
                response = json.loads(row["Response"])
                responses.append({
                    "timestamp": row["Timestamp"],
                    "question": row["Question"],
                    "response": response
                })
            logger.info("Retrieved latest responses", count=len(responses))
            return responses
        except Exception as e:
            logger.error("Failed to retrieve latest responses", error=str(e))
            return []