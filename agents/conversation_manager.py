"""
Enhanced conversation management similar to Claude's approach
"""

import json
import re
from typing import Dict, Any, List
import structlog

logger = structlog.get_logger()


class ConversationManager:
    """Enhanced conversation management similar to Claude's approach"""

    def __init__(self, kql_storage, schema_manager, max_context_pairs=15):
        self.kql_storage = kql_storage
        self.schema_manager = schema_manager
        self.max_context_pairs = max_context_pairs

    async def get_structured_conversation_context(
        self, session_id: str
    ) -> Dict[str, Any]:
        """Get structured conversation context with metadata"""

        try:
            # Enhanced KQL query with metadata extraction
            context_query = f"""
            let recent_messages = (
                ChatHistory_CFO
                | where SessionID == '{session_id}'
                | where Question != 'tables_info' and Question != 'schema_info'
                | order by Timestamp desc
                | take {self.max_context_pairs * 2}
            );
            let important_context = (
                ChatHistory_CFO
                | where SessionID == '{session_id}'
                | where Question != 'tables_info' and Question != 'schema_info'
                | where Question has_any("show", "revenue", "profit", "client", "business unit")
                | order by Timestamp desc
                | take 10
            );
            union recent_messages, important_context
            | distinct Question, Response, Timestamp
            | order by Timestamp asc
            | extend 
                Decoded_Response = case(
                    Response startswith "eyJ" or Response startswith "ew", 
                    base64_decode_tostring(Response),
                    Response
                ),
                IsDataQuery = Question has_any("show", "revenue", "profit", "client", "business unit"),
                IsContextualQuery = Question has_any("why", "how", "what should", "which", "this", "it")
            | project Question, Decoded_Response, Timestamp, IsDataQuery, IsContextualQuery
            """

            result = await self.kql_storage.db_manager.kusto_client.execute(
                self.kql_storage.db_manager.kusto_database, context_query
            )

            # Process into structured format
            conversation_context = {
                "messages": [],
                "last_data_query": None,
                "business_entities_mentioned": set(),
                "filters_in_use": [],
                "session_metadata": {
                    "total_exchanges": 0,
                    "data_queries": 0,
                    "contextual_queries": 0,
                },
            }

            for row in result.primary_results[0] if result.primary_results else []:
                try:
                    response_data = json.loads(row["Decoded_Response"])

                    # Extract business context
                    sql = response_data.get("generated_sql", "")
                    if sql:
                        conversation_context["last_data_query"] = {
                            "question": row["Question"],
                            "sql": sql,
                            "timestamp": row["Timestamp"],
                            "result_count": response_data.get("result_count", 0),
                        }

                        # Extract filters and business entities
                        conversation_context["filters_in_use"].extend(
                            self._extract_sql_filters(sql)
                        )
                        conversation_context["business_entities_mentioned"].update(
                            self._extract_business_entities(sql)
                        )

                    # Build message structure
                    conversation_context["messages"].extend(
                        [
                            {
                                "role": "user",
                                "content": row["Question"],
                                "timestamp": row["Timestamp"],
                                "metadata": {
                                    "is_data_query": row.get("IsDataQuery", False),
                                    "is_contextual": row.get(
                                        "IsContextualQuery", False
                                    ),
                                },
                            },
                            {
                                "role": "assistant",
                                "content": self._format_assistant_message(
                                    response_data
                                ),
                                "timestamp": row["Timestamp"],
                                "metadata": {
                                    "has_sql": bool(sql),
                                    "result_count": response_data.get(
                                        "result_count", 0
                                    ),
                                    "has_visualization": bool(
                                        response_data.get("visualization")
                                    ),
                                },
                            },
                        ]
                    )

                    # Update session stats
                    conversation_context["session_metadata"]["total_exchanges"] += 1
                    if row.get("IsDataQuery"):
                        conversation_context["session_metadata"]["data_queries"] += 1
                    if row.get("IsContextualQuery"):
                        conversation_context["session_metadata"][
                            "contextual_queries"
                        ] += 1

                except json.JSONDecodeError:
                    continue

            return conversation_context

        except Exception as e:
            logger.error("Failed to get structured conversation context", error=str(e))
            return self._empty_context()

    def _extract_sql_filters(self, sql: str) -> List[str]:
        """Extract WHERE conditions from SQL"""
        filters = []
        try:
            if " WHERE " in sql.upper():
                where_part = (
                    sql.upper()
                    .split(" WHERE ")[1]
                    .split(" GROUP BY")[0]
                    .split(" ORDER BY")[0]
                )
                # Simple filter extraction
                if "[Client]" in where_part:
                    client_match = re.search(r"\[Client\]\s*=\s*'([^']+)'", sql)
                    if client_match:
                        filters.append(f"Client = '{client_match.group(1)}'")

                if "DATEPART(YEAR" in where_part:
                    year_match = re.search(
                        r"DATEPART\(YEAR[^)]+\)\s*IN\s*\(([^)]+)\)", sql
                    )
                    if year_match:
                        filters.append(f"Years = {year_match.group(1)}")
        except:
            pass
        return filters

    def _extract_business_entities(self, sql: str) -> set[str]:
        """Extract business entities from SQL"""
        entities = set()
        entity_patterns = [
            "[Client]",
            "[Region]",
            "[Country]",
            "[Business Unit]",
            "[Client Tier]",
        ]
        for pattern in entity_patterns:
            if pattern in sql:
                entities.add(pattern.strip("[]"))
        return entities

    def _format_assistant_message(self, response_data: Dict) -> str:
        """Format assistant response for context"""
        parts = []
        if response_data.get("generated_sql"):
            parts.append(f"SQL: {response_data['generated_sql'][:100]}...")

        result_count = response_data.get("result_count", 0)
        if result_count > 0:
            parts.append(f"Found {result_count} records")

        if response_data.get("analysis"):
            parts.append(f"Analysis: {response_data['analysis'][:200]}...")

        return " | ".join(parts)

    def _empty_context(self):
        """Return empty context structure"""
        return {
            "messages": [],
            "last_data_query": None,
            "business_entities_mentioned": set(),
            "filters_in_use": [],
            "session_metadata": {
                "total_exchanges": 0,
                "data_queries": 0,
                "contextual_queries": 0,
            },
        }
