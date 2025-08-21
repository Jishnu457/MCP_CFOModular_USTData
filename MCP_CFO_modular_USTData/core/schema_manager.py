"""
Schema management with caching
"""
import asyncio
import time
from typing import List, Dict, Any
import structlog

logger = structlog.get_logger()

class SchemaManager:
    """Centralized schema management with caching"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.cached_tables_info = None
        self.schema_cache_timestamp = None
        self.schema_cache_duration = 3600  # Cache for 1 hour
    
    async def get_cached_tables_info(self):
        """Get schema from memory cache - NO KQL storage for schema"""
        current_time = time.time()
        
        # Check if we have valid cached data
        if (self.cached_tables_info is not None and 
            self.schema_cache_timestamp is not None and 
            (current_time - self.schema_cache_timestamp) < self.schema_cache_duration):
            logger.info("Schema cache hit from memory")
            return self.cached_tables_info
        
        # Cache is empty or expired, fetch fresh data
        logger.info("Fetching fresh schema data")
        start_time = time.time()
        
        try:
            tables_info = await self.get_tables_info()
            
            # Cache in memory only - NOT in KQL
            self.cached_tables_info = tables_info
            self.schema_cache_timestamp = current_time
            
            duration = time.time() - start_time
            logger.info("Schema fetched and cached in memory", 
                       duration=duration, 
                       table_count=len(tables_info))
            
            return tables_info
            
        except Exception as e:
            logger.error("Failed to fetch schema", error=str(e))
            # Return cached data if available, even if expired
            if self.cached_tables_info is not None:
                logger.warning("Using expired schema cache due to fetch error")
                return self.cached_tables_info
            raise
    
    async def get_tables_info(self):
        """Get detailed table information"""
        loop = asyncio.get_event_loop()
        query = """
        SELECT TABLE_SCHEMA, TABLE_NAME, TABLE_TYPE
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
        AND TABLE_SCHEMA NOT IN ('sys', 'INFORMATION_SCHEMA')
        ORDER BY TABLE_SCHEMA, TABLE_NAME
        """
        tables = await loop.run_in_executor(None, lambda: self.db_manager.execute_sql_query(query))
        tables_info = []

        async def fetch_table_metadata(table):
            column_query = """
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
            ORDER BY ORDINAL_POSITION
            """
            fk_query = """
            SELECT
                C.CONSTRAINT_NAME,
                C.TABLE_NAME,
                C.COLUMN_NAME,
                R.TABLE_NAME AS REFERENCED_TABLE,
                R.COLUMN_NAME AS REFERENCED_COLUMN
            FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS RC
            JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE C
                ON C.CONSTRAINT_NAME = RC.CONSTRAINT_NAME
            JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE R
                ON R.CONSTRAINT_NAME = RC.UNIQUE_CONSTRAINT_NAME
            WHERE C.TABLE_SCHEMA = :schema AND C.TABLE_NAME = :table
            """
            sample_query = f"SELECT TOP 3 * FROM [{table['TABLE_SCHEMA']}].[{table['TABLE_NAME']}]"
            
            columns, fks, sample_data = await asyncio.gather(
                loop.run_in_executor(None, lambda: self.db_manager.execute_sql_query(column_query, {"schema": table["TABLE_SCHEMA"], "table": table["TABLE_NAME"]})),
                loop.run_in_executor(None, lambda: self.db_manager.execute_sql_query(fk_query, {"schema": table["TABLE_SCHEMA"], "table": table["TABLE_NAME"]})),
                loop.run_in_executor(None, lambda: self.db_manager.execute_sql_query(sample_query)),
                return_exceptions=True
            )
            
            if isinstance(columns, Exception) or isinstance(fks, Exception) or isinstance(sample_data, Exception):
                logger.warning("Failed to fetch metadata for table", table=table["TABLE_NAME"], error=str(columns or fks or sample_data))
                return None
            
            # Process table metadata
            fk_info = [f"{fk['COLUMN_NAME']} references {fk['REFERENCED_TABLE']}.{fk['REFERENCED_COLUMN']}" for fk in fks]
            enhanced_columns = []
            numeric_columns = []
            text_columns = []
            date_columns = []
            column_values = {}
            
            for col in columns:
                col_name = col['COLUMN_NAME']
                data_type = col['DATA_TYPE'].lower()
                nullable = 'Nullable' if col['IS_NULLABLE'] == 'YES' else 'Not Nullable'
                
                if data_type in ['int', 'bigint', 'smallint', 'tinyint', 'decimal', 'numeric', 'float', 'real', 'money', 'smallmoney']:
                    numeric_columns.append(col_name)
                    enhanced_columns.append(f"[{col_name}] ({data_type.upper()}, {nullable}) - NUMERIC: Use AVG(), SUM(), MAX(), MIN()")
                elif data_type in ['varchar', 'nvarchar', 'char', 'nchar', 'text', 'ntext']:
                    text_columns.append(col_name)
                    enhanced_columns.append(f"[{col_name}] ({data_type.upper()}, {nullable}) - TEXT: Use COUNT(), CASE statements, GROUP BY - NEVER AVG()")
                    try:
                        distinct_query = f"SELECT DISTINCT TOP 10 [{col_name}] FROM [{table['TABLE_SCHEMA']}].[{table['TABLE_NAME']}] WHERE [{col_name}] IS NOT NULL"
                        distinct_values = await loop.run_in_executor(None, lambda: self.db_manager.execute_sql_query(distinct_query))
                        column_values[col_name] = [row[col_name] for row in distinct_values]
                    except:
                        column_values[col_name] = []
                elif data_type in ['datetime', 'datetime2', 'date', 'time', 'datetimeoffset', 'smalldatetime']:
                    date_columns.append(col_name)
                    enhanced_columns.append(f"[{col_name}] ({data_type.upper()}, {nullable}) - DATE: Use MAX(), MIN(), date functions")
                else:
                    enhanced_columns.append(f"[{col_name}] ({data_type.upper()}, {nullable})")
            
            return {
                "table": f"[{table['TABLE_SCHEMA']}].[{table['TABLE_NAME']}]",
                "columns": enhanced_columns,
                "numeric_columns": numeric_columns,
                "text_columns": text_columns,
                "date_columns": date_columns,
                "foreign_keys": fk_info,
                "sample_data": sample_data[:2] if sample_data else [],
                "column_values": column_values
            }

        tables_info = await asyncio.gather(*(fetch_table_metadata(table) for table in tables))
        return [info for info in tables_info if info]
    
    async def preload_schema(self):
        """Preload schema during application startup"""
        try:
            logger.info("Preloading database schema...")
            start_time = time.time()
            
            tables_info = await self.get_cached_tables_info()
            
            duration = time.time() - start_time
            logger.info("Schema preloaded successfully", 
                       duration=duration,
                       table_count=len(tables_info),
                       total_columns=sum(len(t.get('columns', [])) for t in tables_info))
            
            return True
            
        except Exception as e:
            logger.error("Schema preload failed", error=str(e))
            print(f"⚠️  Schema preload failed: {e}")
            print("    App will still work, but first query may be slower")
            return False
    
    def refresh_cache(self):
        """Manually refresh the schema cache"""
        self.cached_tables_info = None
        self.schema_cache_timestamp = None