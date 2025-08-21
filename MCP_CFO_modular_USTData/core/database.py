"""
Database connection management for SQL and KQL
"""
import asyncio
from typing import List, Dict, Any
from datetime import datetime
from decimal import Decimal
import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder
from azure.kusto.data.exceptions import KustoServiceError

from config.settings import ConfigManager
from utils.helpers import Utils

logger = structlog.get_logger()

class DatabaseManager:
    """Centralized database connection management"""
    
    def __init__(self):
        self.kusto_client = None
        self.kusto_database = None
        self.sql_engine = None
        self.setup_connections()
    
    def setup_connections(self):
        """Initialize both KQL and SQL connections"""
        ConfigManager.validate_environment()
        self.setup_kql_client()
        self.setup_sql_engine()
    
    def setup_kql_client(self):
        """Initialize KQL client with proper error handling"""
        try:
            config = ConfigManager.get_database_config()
            
            # Log configuration (without secrets)
            logger.info("KQL configuration loaded", 
                       cluster=config["kusto_cluster"],
                       database=config["kusto_database"],
                       tenant_id=config["tenant_id"],
                       client_id=config["client_id"],
                       has_secret=bool(config["client_secret"]))
            
            # Validate KQL cluster URI format
            if not config["kusto_cluster"].startswith("https://"):
                raise ValueError(f"KUSTO_CLUSTER must start with 'https://'. Got: {config['kusto_cluster']}")
                
            if "ingest" in config["kusto_cluster"].lower():
                logger.warning("KUSTO_CLUSTER appears to be an ingestion endpoint", cluster=config["kusto_cluster"])
                raise ValueError(
                    f"KUSTO_CLUSTER should be a query endpoint, not ingestion. "
                    f"Expected format: https://<eventhouse>.<region>.kusto.fabric.microsoft.com"
                )
            
            # Initialize KQL connection
            kusto_connection_string = KustoConnectionStringBuilder.with_aad_application_key_authentication(
                config["kusto_cluster"],
                config["client_id"], 
                config["client_secret"],
                config["tenant_id"]
            )
            
            self.kusto_client = KustoClient(kusto_connection_string)
            self.kusto_database = config["kusto_database"]
            logger.info("KQL client initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize KQL client", error=str(e))
            raise RuntimeError(f"KQL client initialization failed: {str(e)}")
    
    def setup_sql_engine(self):
        """Initialize SQL database connection"""
        config = ConfigManager.get_database_config()
        
        connection_string = (
            f"Driver={{ODBC Driver 17 for SQL Server}};"
            f"Server={config['sql_endpoint']};"
            f"Database={config['database']};"
            f"Authentication=ActiveDirectoryServicePrincipal;"
            f"UID={config['client_id']};"
            f"PWD={config['client_secret']};"
            f"Encrypt=yes;"
            f"TrustServerCertificate=no;"
        )
        
        self.sql_engine = create_engine(
            f"mssql+pyodbc:///?odbc_connect={connection_string}",
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600
        )
    
    def execute_sql_query(self, query: str, params=None) -> List[Dict[str, Any]]:
        """Enhanced execute_query with better GROUP BY error messages and number formatting"""
        try:
            with self.sql_engine.connect() as conn:
                executable_query = text(query)
                cursor = conn.execute(executable_query, params or {})
                columns = cursor.keys()
                rows = cursor.fetchall()
                result = []
                
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    for key, value in row_dict.items():
                        if isinstance(value, datetime):
                            row_dict[key] = value.isoformat()
                        elif hasattr(value, 'date') and callable(getattr(value, 'date')):
                            row_dict[key] = value.isoformat()
                        elif isinstance(value, (bytes, bytearray)):
                            row_dict[key] = value.decode('utf-8', errors='ignore')
                        elif isinstance(value, Decimal):
                            row_dict[key] = Utils.format_number(value, 2)
                        elif isinstance(value, float):
                            row_dict[key] = Utils.format_number(value, 2)
                    result.append(row_dict)
                
                # Apply formatting to all numeric results
                formatted_result = Utils.format_results_data(result, 2)
                return formatted_result
                
        except Exception as e:
            error_str = str(e)
            logger.error("Query execution error", query=query, params=params, error=str(e))
            
            # Improved error handling for GROUP BY issues
            if "8120" in error_str or "GROUP BY" in error_str:
                if "is not contained in either an aggregate function or the GROUP BY clause" in error_str:
                    raise Exception(
                        "SQL GROUP BY error: All non-aggregate columns in SELECT must be included in GROUP BY clause. "
                        "Fix: Add missing columns to GROUP BY, or use aggregate functions like COUNT(), SUM(), AVG() for calculated fields. "
                        f"Original error: {error_str}"
                    )
                else:
                    raise Exception(
                        "SQL GROUP BY error: When using GROUP BY, all SELECT columns must either be in the GROUP BY clause "
                        "or use aggregate functions (COUNT, SUM, AVG, etc.). "
                        f"Original error: {error_str}"
                    )
            else:
                raise
    
    async def test_kql_connection(self):
        """Test the KQL connection with a simple query"""
        try:
            test_query = "print 'KQL connection test successful'"
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.kusto_client.execute(self.kusto_database, test_query)
            )
            logger.info("KQL connection test passed")
            return True
        except Exception as e:
            logger.error("KQL connection test failed", error=str(e))
            return False