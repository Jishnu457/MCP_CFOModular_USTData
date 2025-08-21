"""
AI service management and integrations
"""
import structlog
from openai import AsyncAzureOpenAI
from azure.identity import DefaultAzureCredential, ClientSecretCredential
from azure.ai.projects import AIProjectClient

from config.settings import ConfigManager

# Check for optional imports
try:
    from msgraph import GraphServiceClient
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False

logger = structlog.get_logger()

class AIServiceManager:
    """Consolidated AI service management"""
    
    def __init__(self):
        self.ai_foundry_enabled = False
        self.project_client = None
        self.graph_client = None
        self.intelligent_agent = None
        self.openai_client = None
        self.setup_services()
    
    def setup_services(self):
        """Initialize all AI services"""
        self.setup_openai_client()
        self.setup_ai_foundry()
        if GRAPH_AVAILABLE:
            self.setup_graph_client()
    
    def setup_openai_client(self):
        """Initialize Azure OpenAI client"""
        config = ConfigManager.get_ai_config()
        
        self.openai_client = AsyncAzureOpenAI(
            api_key=config["openai_api_key"],
            api_version=config["openai_api_version"],
            azure_endpoint=config["openai_endpoint"]
        )
    
    def setup_ai_foundry(self):
        """Initialize Azure AI Foundry client"""
        try:
            config = ConfigManager.get_ai_config()
            project_endpoint = config["ai_project_endpoint"]
            
            if not project_endpoint:
                logger.info("AI_PROJECT_ENDPOINT not set, AI Foundry features disabled")
                self.intelligent_agent = None
                self.ai_foundry_enabled = False
                return False
                
            credential = DefaultAzureCredential()
            
            self.project_client = AIProjectClient(
                endpoint=project_endpoint,
                credential=credential
            )
            
            logger.info("Azure AI Foundry client initialized successfully")
            self.ai_foundry_enabled = True
            
            try:
                # Import here to avoid circular imports
                from agents.intelligent_agent import IntelligentAnalyticsAgent
                self.intelligent_agent = IntelligentAnalyticsAgent(self.project_client)
                logger.info("Intelligent analytics agent initialized successfully")
            except Exception as agent_error:
                logger.error("Failed to initialize intelligent agent", error=str(agent_error))
                self.intelligent_agent = None
            
            return True
            
        except Exception as e:
            logger.warning("Azure AI Foundry setup failed", error=str(e))
            logger.info("Continuing with standard OpenAI integration")
            self.intelligent_agent = None
            return False
    
    def setup_graph_client(self):
        """Initialize Microsoft Graph client for email"""
        try:
            config = ConfigManager.get_ai_config()
            
            tenant_id = config["graph_tenant_id"]
            client_id = config["graph_client_id"]
            client_secret = config["graph_client_secret"]
            
            if not all([tenant_id, client_id, client_secret]):
                logger.info("Graph API credentials not complete, email features disabled")
                return False
                
            credential = ClientSecretCredential(
                tenant_id=tenant_id,
                client_id=client_id,
                client_secret=client_secret
            )
            
            self.graph_client = GraphServiceClient(credential)
            logger.info("Microsoft Graph client initialized successfully")
            return True
            
        except Exception as e:
            logger.warning("Microsoft Graph setup failed", error=str(e))
            return False
    
    async def ask_intelligent_llm_async(self, prompt: str) -> str:
        """Ask LLM with consolidated error handling"""
        config = ConfigManager.get_ai_config()
        deployment = config["openai_deployment"]
        
        if not deployment:
            raise ValueError("AZURE_OPENAI_DEPLOYMENT not set")
            
        try:
            response = await self.openai_client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": "You are a helpful, friendly AI assistant with expertise in data analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=15000,
                seed=42
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error("LLM request failed", error=str(e))
            raise