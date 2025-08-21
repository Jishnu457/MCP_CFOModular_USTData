"""
Main FastAPI application entry point
"""

import uvicorn
from fastapi import FastAPI

# Configuration and logging
from config.logging_config import setup_logging
from config.settings import ConfigManager, AppSettings

# Core services
from core.database import DatabaseManager
from core.kql_storage import KQLStorage
from core.schema_manager import SchemaManager

# AI and business services
from services.ai_services import AIServiceManager
from services.email_service import EmailService
from services.sharepoint_service import SharePointUploader
from services.prompt_manager import PromptManager
from services.visualization import VisualizationManager
from services.report_generator import ReportGenerator
from services.response_formatter import ResponseFormatter, SmartResponseEnhancer

# Agents
from agents.conversation_manager import ConversationManager

# API setup
from api.middleware import setup_middleware

# Initialize logging
logger = setup_logging()

# Initialize core services
db_manager = DatabaseManager()
kql_storage = KQLStorage(db_manager)
schema_manager = SchemaManager(db_manager)

# Initialize AI services
ai_services = AIServiceManager()
email_service = EmailService(ai_services.graph_client)
sharepoint_uploader = SharePointUploader()
prompt_manager = PromptManager(ai_services)
viz_manager = VisualizationManager(ai_services)
report_generator = ReportGenerator()
response_formatter = ResponseFormatter(ai_services)
response_enhancer = SmartResponseEnhancer(ai_services)

# Set AI services for report generator (dependency injection)
report_generator.set_ai_services(ai_services)

# Initialize conversation manager
conversation_manager = ConversationManager(kql_storage, schema_manager)

# Initialize analytics engine (import here to avoid circular imports)
from services.analytics_engine import AnalyticsEngine

analytics_engine = AnalyticsEngine(
    db_manager, schema_manager, kql_storage, ai_services, viz_manager, prompt_manager
)

# Create FastAPI app
app = FastAPI(
    title=AppSettings.TITLE,
    description=AppSettings.DESCRIPTION,
    version=AppSettings.VERSION,
)

# Setup middleware
setup_middleware(app)

# Import and configure API routes with dependency injection
from api.endpoints import analytics, chat, admin, health


# Configure routes with dependencies
def configure_routes():
    """Configure API routes with proper dependencies"""

    # Inject dependencies into endpoint modules
    analytics.analytics_engine = analytics_engine
    analytics.ai_services = ai_services
    analytics.email_service = email_service
    analytics.report_generator = report_generator

    chat.kql_storage = kql_storage
    chat.db_manager = db_manager

    admin.db_manager = db_manager
    admin.kql_storage = kql_storage
    admin.schema_manager = schema_manager

    health.db_manager = db_manager
    health.schema_manager = schema_manager

    # Include routers
    app.include_router(analytics.router, prefix="/api", tags=["Analytics"])
    app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
    app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])
    app.include_router(health.router, tags=["Health"])


# Configure routes
configure_routes()


@app.on_event("startup")
async def startup_event():
    """Enhanced startup with AI Foundry and schema preloading"""
    try:
        logger.info("Starting enhanced application initialization...")

        # Test connections
        kql_ok = await db_manager.test_kql_connection()
        if not kql_ok:
            logger.warning("KQL connection failed during startup")
        else:
            logger.info("KQL connection test passed")

        await kql_storage.initialize_kql_table()

        schema_preloaded = await schema_manager.preload_schema()
        if schema_preloaded:
            print("Schema preloaded - first query will be fast!")
        else:
            print(" Schema preload failed - first query may be slower")

        # Check additional services
        if ai_services.ai_foundry_enabled:
            print(" Azure AI Foundry agents initialized")
        else:
            print("Azure AI Foundry not available - using standard OpenAI only")

        if ai_services.graph_client:
            print(" Microsoft Graph email service available")
        else:
            print(
                " Email service not available - configure Graph API for email features"
            )

        # Check report generation capability
        try:
            from reportlab.pdfgen import canvas

            print("Report generation libraries available")
        except ImportError:
            print(" Report generation not available - install reportlab and xlsxwriter")

        logger.info("Enhanced application startup completed successfully")

    except Exception as e:
        logger.error("Enhanced startup failed", error=str(e))
        print(f"Startup Error: {e}")


if __name__ == "__main__":
    print(" Intelligent SQL Analytics Assistant")
    print(" Powered by Microsoft Fabric SQL Database and KQL Storage")
    print("Advanced analytics engine")
    print(" Smart visualization")
    print("")
    print(" Key Features:")
    print("• Natural language queries")
    print("• Automatic SQL generation")
    print("• Business-oriented insights")
    print("• Context-aware visualizations")
    print("• KQL-based conversation history")
    print("• AI-powered analysis")
    print("• Email notifications")
    print("• Professional report generation")
    print("")
    print(" Example Questions:")
    print("• 'What is the average cyber risk score?'")
    print("• 'Show critical vulnerabilities (CVSS ≥ 7.0)'")
    print("• 'How many unpatched devices by type?'")
    print("• 'Show trends in incidents over time'")
    print("• 'What are their departments?'")
    print("")

    try:
        ConfigManager.validate_environment()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        print(f" Failed to start application: {e}")
        exit(1)
