"""
Configuration management for Intelligent Fabric Analytics
"""
import os
from typing import List
import structlog
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = structlog.get_logger()

class ConfigManager:
    """Centralized configuration management"""
    
    REQUIRED_VARS = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT", 
        "AZURE_OPENAI_DEPLOYMENT",
        "FABRIC_SQL_ENDPOINT",
        "FABRIC_DATABASE",
        "FABRIC_CLIENT_ID",
        "FABRIC_CLIENT_SECRET",
        "KUSTO_CLUSTER",
        "KUSTO_DATABASE", 
        "FABRIC_TENANT_ID"
    ]
    
    OPTIONAL_AI_VARS = [
        "AI_PROJECT_ENDPOINT",
        "GRAPH_CLIENT_ID",
        "GRAPH_CLIENT_SECRET",
        "GRAPH_TENANT_ID"
    ]
    
    OPTIONAL_SHAREPOINT_VARS = [
        "SHAREPOINT_TENANT_ID",
        "SHAREPOINT_CLIENT_ID",
        "SHAREPOINT_CLIENT_SECRET",
        "SHAREPOINT_SITE_ID",
        "SHAREPOINT_DOCUMENT_LIBRARY_ID"
    ]
    
    @classmethod
    def validate_environment(cls):
        """Validate all required environment variables"""
        missing = []
        for var in cls.REQUIRED_VARS:
            value = os.getenv(var)
            if not value or value.strip() == "":
                missing.append(var)
                
        if missing:
            logger.error("Missing environment variables", missing=missing)
            raise RuntimeError(f"Missing required environment variables: {missing}")
        
        ai_vars_available = all(os.getenv(var) for var in cls.OPTIONAL_AI_VARS)
        sharepoint_vars_available = all(os.getenv(var) for var in cls.OPTIONAL_SHAREPOINT_VARS)
        
        logger.info("Environment validation passed", 
                    total_vars=len(cls.REQUIRED_VARS),
                    ai_foundry_enabled=ai_vars_available,
                    sharepoint_enabled=sharepoint_vars_available)
        return True

    @classmethod
    def get_database_config(cls):
        """Get database configuration"""
        return {
            "sql_endpoint": os.getenv("FABRIC_SQL_ENDPOINT"),
            "database": os.getenv("FABRIC_DATABASE"),
            "client_id": os.getenv("FABRIC_CLIENT_ID"),
            "client_secret": os.getenv("FABRIC_CLIENT_SECRET"),
            "tenant_id": os.getenv("FABRIC_TENANT_ID"),
            "kusto_cluster": os.getenv("KUSTO_CLUSTER"),
            "kusto_database": os.getenv("KUSTO_DATABASE")
        }
    
    @classmethod
    def get_ai_config(cls):
        """Get AI service configuration"""
        return {
            "openai_api_key": os.getenv("AZURE_OPENAI_API_KEY"),
            "openai_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "openai_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            "openai_api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            "ai_project_endpoint": os.getenv("AI_PROJECT_ENDPOINT"),
            "graph_client_id": os.getenv("GRAPH_CLIENT_ID"),
            "graph_client_secret": os.getenv("GRAPH_CLIENT_SECRET"),
            "graph_tenant_id": os.getenv("GRAPH_TENANT_ID")
        }
    
    @classmethod
    def get_sharepoint_config(cls):
        """Get SharePoint configuration"""
        return {
            "tenant_id": os.getenv("SHAREPOINT_TENANT_ID") or os.getenv("FABRIC_TENANT_ID"),
            "client_id": os.getenv("SHAREPOINT_CLIENT_ID") or os.getenv("FABRIC_CLIENT_ID"),
            "client_secret": os.getenv("SHAREPOINT_CLIENT_SECRET") or os.getenv("FABRIC_CLIENT_SECRET"),
            "site_id": os.getenv("SHAREPOINT_SITE_ID"),
            "document_library_id": os.getenv("SHAREPOINT_DOCUMENT_LIBRARY_ID"),
            "scope": "https://graph.microsoft.com/.default"
        }

class AppSettings:
    """Application settings"""
    
    # API Settings
    TITLE = "Intelligent Microsoft Fabric SQL Analytics"
    DESCRIPTION = "Processes natural language questions to generate SQL queries, execute them, and provide insights with optional visualizations."
    VERSION = "1.0.0"
    
    # CORS Settings
    CORS_ORIGINS = ["*"]
    CORS_CREDENTIALS = True
    CORS_METHODS = ["*"]
    CORS_HEADERS = ["*"]
    
    # Rate Limiting
    RATE_LIMIT = "10/minute"
    
    # Cache Settings
    SCHEMA_CACHE_DURATION = 3600  # 1 hour
    
    # Default Session Settings
    DEFAULT_SESSION_PREFIX = "powerbi_"
    DEFAULT_SESSION_FALLBACK = "default-session-1234567890"