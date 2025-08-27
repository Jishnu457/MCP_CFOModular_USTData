"""
Intelligent analytics agent for advanced AI capabilities
"""
import asyncio
import json
import time
from typing import Dict, List, Any, Optional
import structlog

logger = structlog.get_logger()


class IntelligentAnalyticsAgent:
    """Enhanced agent for data analytics with AI capabilities"""

    def __init__(self, project_client):
        self.project_client = project_client
        self.data_agent_id = None
        self.report_agent_id = None
        self.email_agent_id = None

        if project_client:
            self.setup_agents()

    def setup_agents(self):
        """Create specialized agents for different tasks"""
        try:
            self.data_agent_id = self.create_data_agent()
            self.report_agent_id = self.create_report_agent()
            self.email_agent_id = self.create_email_agent()
            logger.info("All AI agents created successfully")

        except Exception as e:
            logger.error("Failed to setup AI agents", error=str(e))

    def create_data_agent(self):
        """Create agent specialized in data analysis"""
        try:
            agent = self.project_client.agents.create_agent(
                model="gpt-4o",
                name="data-analyst-agent",
                instructions="""You are an expert data analyst specializing in cybersecurity and business intelligence.
                
                Your capabilities:
                - Analyze SQL query results and identify patterns
                - Provide insights into cybersecurity metrics
                - Explain trends and anomalies in data
                - Suggest actionable recommendations
                - Format analysis in clear, business-friendly language
                
                Always provide:
                1. Key findings summary
                2. Detailed insights with numbers
                3. Business implications
                4. Actionable recommendations
                """,
                tools=[],
            )
            return agent.id
        except Exception as e:
            logger.error("Failed to create data agent", error=str(e))
            return None

    def create_report_agent(self):
        """Create agent specialized in report generation"""
        try:
            agent = self.project_client.agents.create_agent(
                model="gpt-4o-mini",
                name="report-generator-agent",
                instructions="""You are a professional report writer specializing in executive summaries and data reports.
                
                Your capabilities:
                - Generate executive summaries from data analysis
                - Create structured reports with clear sections
                - Write professional, concise content
                - Format content suitable for PDF/Excel reports
                - Include key metrics and recommendations
                
                Report structure should include:
                1. Executive Summary
                2. Key Metrics
                3. Detailed Findings
                4. Recommendations
                5. Next Steps
                """,
                tools=[],
            )
            return agent.id
        except Exception as e:
            logger.error("Failed to create report agent", error=str(e))
            return None

    def create_email_agent(self):
        """Create agent specialized in email communications"""
        try:
            agent = self.project_client.agents.create_agent(
                model="gpt-4o-mini",
                name="email-agent",
                instructions="""You are a professional communication specialist for sending analytics reports.
                
                Your capabilities:
                - Write professional email content
                - Create compelling subject lines
                - Format emails appropriately for business context
                - Include proper context and next steps
                
                Email should be:
                - Professional and concise
                - Include key highlights from the report
                - Provide context for recipients
                - Include clear call-to-action if needed
                """,
                tools=[],
            )
            return agent.id
        except Exception as e:
            logger.error("Failed to create email agent", error=str(e))
            return None

    async def analyze_with_ai(self, data, question, context=None):
        """Use AI agent to analyze data and provide insights"""
        if not self.project_client or not self.data_agent_id:
            return None

        try:
            thread = self.project_client.agents.create_thread()

            analysis_prompt = f"""
            Original Question: {question}
            
            Data Results: {json.dumps(data[:10], default=str)}
            Total Records: {len(data)}
            
            {f"Additional Context: {context}" if context else ""}
            
            Please provide a comprehensive analysis of this data including key insights, trends, and actionable recommendations.
            """

            message = self.project_client.agents.create_message(
                thread_id=thread.id, role="user", content=analysis_prompt
            )

            run = self.project_client.agents.create_run(
                thread_id=thread.id, assistant_id=self.data_agent_id
            )

            response = await self.wait_for_run_completion(thread.id, run.id)
            return response

        except Exception as e:
            logger.error("AI analysis failed", error=str(e))
            return None

    async def wait_for_run_completion(self, thread_id, run_id, timeout=60):
        """Wait for agent run to complete and return response"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                run = self.project_client.agents.get_run(
                    thread_id=thread_id, run_id=run_id
                )

                if run.status == "completed":
                    messages = self.project_client.agents.list_messages(
                        thread_id=thread_id
                    )
                    if messages.data:
                        latest_message = messages.data[0]
                        if latest_message.role == "assistant":
                            return latest_message.content[0].text.value
                    break
                elif run.status in ["failed", "cancelled", "expired"]:
                    logger.error(f"Agent run failed with status: {run.status}")
                    break

                await asyncio.sleep(1)

            except Exception as e:
                logger.error("Error waiting for run completion", error=str(e))
                break

        return None
