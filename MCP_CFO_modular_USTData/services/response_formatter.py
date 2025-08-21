"""
Natural response formatter for conversational AI-style responses
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import structlog

logger = structlog.get_logger()


class ResponseFormatter:
    """Format responses in a natural, conversational manner similar to Claude/ChatGPT"""

    def __init__(self, ai_services):
        self.ai_services = ai_services

    async def format_response(
        self, question: str, raw_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format the response naturally based on the question type and content"""

        try:
            # Determine response style based on question
            response_style = self._determine_response_style(question, raw_result)

            # Format based on style
            if response_style == "conversational":
                return await self._format_conversational(question, raw_result)
            elif response_style == "data_analysis":
                return await self._format_data_analysis(question, raw_result)
            elif response_style == "error_helpful":
                return await self._format_helpful_error(question, raw_result)
            elif response_style == "greeting":
                return await self._format_greeting(question, raw_result)
            else:
                return await self._format_default(question, raw_result)

        except Exception as e:
            logger.error("Response formatting failed", error=str(e))
            return await self._format_fallback(question, raw_result)

    def _determine_response_style(
        self, question: str, raw_result: Dict[str, Any]
    ) -> str:
        """Determine the appropriate response style"""

        question_lower = question.lower().strip()

        # Check for greetings
        if question_lower in ["hi", "hello", "hey", "greetings"]:
            return "greeting"

        # Check for errors
        if "error" in raw_result or raw_result.get("response_type") == "error":
            return "error_helpful"

        # Check for conversational questions
        if raw_result.get("response_type") == "conversational":
            return "conversational"

        # Check for data analysis
        if raw_result.get("generated_sql") or raw_result.get("result_count", 0) > 0:
            return "data_analysis"

        return "default"

    async def _format_conversational(
        self, question: str, raw_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format conversational responses with simple highlighting"""

        analysis = raw_result.get("analysis", "I'd be happy to help with that.")

        # Simple prompt for conversational highlighting
        prompt = f"""
        User asked: "{question}"
        Current response: "{analysis}"
        
        Make this more engaging by:
        1. Use **bold** for key capabilities and important points
        2. Be warm and helpful
        3. Highlight what the user can do next
        4. Keep it natural and conversational
        """

        try:
            conversational_response = await self.ai_services.ask_intelligent_llm_async(
                prompt
            )
        except:
            conversational_response = f"**I'd be happy to help!** {analysis}"

        return {
            "message": conversational_response,
            "type": "conversational",
            "suggestions": [
                "Try asking: **'What data do I have?'**",
                "Get analysis: **'Show me revenue for 2024'**",
                "Find insights: **'What are the key trends?'**",
            ],
            "session_id": raw_result.get("session_id"),
            "timestamp": datetime.now().isoformat(),
        }

    async def _format_data_analysis(
        self, question: str, raw_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format data analysis responses with simple highlighting"""

        result_count = raw_result.get("result_count", 0)
        analysis = raw_result.get("analysis", "")
        enhanced_analysis = raw_result.get("enhanced_analysis", "")
        generated_sql = raw_result.get("generated_sql", "")

        # Simple prompt that just asks for highlighting
        prompt = f"""
        The user asked: "{question}"
        
        I found {result_count} records in the database.
        
        Analysis: {enhanced_analysis or analysis}
        
        Rewrite this response following these guidelines:
        1. Use **bold** for important numbers and key findings
        2. Start with a natural, conversational opening sentence
        3. Highlight 2-3 main insights with **bold**
        4. Use proper line breaks between sections
        5. Professional tone but conversational language
        6. NO phrases like "Here's a friendly version" or "more engaging response"
        7. NO mentions of "conversational" or "reader-friendly" 
        8. Get straight to the business insights
        9. End with what this means or suggest next steps

        Start like you're naturally sharing interesting findings with a colleague - conversational but professional.
        """

        try:
            natural_response = await self.ai_services.ask_intelligent_llm_async(prompt)
        except:
            # Simple fallback - just add bold to numbers
            natural_response = f"I analyzed your question about {question.lower()} and found **{result_count} records**. {enhanced_analysis or analysis}"

        # Rest of your existing code stays the same
        response = {
            "message": natural_response,
            "type": "analysis",
            "found_records": result_count,
            "session_id": raw_result.get("session_id"),
            "timestamp": datetime.now().isoformat(),
        }

        # Add data if requested or relevant
        if result_count > 0 and result_count <= 10:
            response["data"] = raw_result.get("sample_data", [])
        elif result_count > 10:
            response["data_sample"] = raw_result.get("sample_data", [])[:5]
            response["note"] = (
                f"Showing 5 of **{result_count}** records. Would you like to see more or filter the results?"
            )

        # Add visualization if available
        if raw_result.get("visualization"):
            response["chart"] = raw_result["visualization"]

        return response

    async def _format_helpful_error(
        self, question: str, raw_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format errors in a helpful, non-technical way"""

        error = raw_result.get("error", "I encountered an issue")
        suggestion = raw_result.get("suggestion", "Try rephrasing your question")

        prompt = f"""
        User asked: "{question}"
        
        There was an issue: {error}
        
        Rewrite this as a helpful, encouraging response like Claude would give:
        - Don't mention technical errors
        - Be supportive and solution-focused
        - Offer specific alternatives
        - Acknowledge what they were trying to do
        - Guide them toward success
        - Use encouraging, friendly tone
        """

        try:
            helpful_response = await self.ai_services.ask_intelligent_llm_async(prompt)
        except:
            helpful_response = f"I understand you're asking about {question.lower()}. Let me help you get the information you need. {suggestion}"

        return {
            "message": helpful_response,
            "type": "help",
            "suggestions": [
                "Try being more specific: 'Show me sales data for Q1 2024'",
                "Ask about available data: 'What information do you have?'",
                "Request examples: 'Give me some example questions'",
            ],
            "session_id": raw_result.get("session_id"),
            "timestamp": datetime.now().isoformat(),
        }

    async def _format_greeting(
        self, question: str, raw_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format greeting responses warmly"""

        greeting_response = f"""
        Hello! I'm your intelligent analytics assistant. I can help you explore and understand your data using natural language.
        
        I can assist you with:
        - Analyzing your financial data and business metrics
        - Creating visualizations and reports
        - Finding trends and insights in your data
        - Answering specific questions about your business performance
        
        What would you like to explore today?
        """

        return {
            "message": greeting_response,
            "type": "greeting",
            "quick_starts": [
                "What data do I have available?",
                "Show me revenue trends for this year",
                "Create a profit and loss summary",
                "What are my top performing products?",
            ],
            "session_id": raw_result.get("session_id"),
            "timestamp": datetime.now().isoformat(),
        }

    async def _format_default(
        self, question: str, raw_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Default formatting for any response"""

        analysis = raw_result.get("analysis", "I've processed your request.")

        return {
            "message": analysis,
            "type": "response",
            "session_id": raw_result.get("session_id"),
            "timestamp": datetime.now().isoformat(),
            "raw_data": raw_result if len(str(raw_result)) < 1000 else None,
        }

    async def _format_fallback(
        self, question: str, raw_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback formatting if all else fails"""

        return {
            "message": f"I understand you're asking about {question}. I'm working on getting you the best answer possible.",
            "type": "processing",
            "session_id": raw_result.get("session_id"),
            "timestamp": datetime.now().isoformat(),
        }


class SmartResponseEnhancer:
    """Enhance responses with context-aware information"""

    def __init__(self, ai_services):
        self.ai_services = ai_services

    async def enhance_with_context(
        self,
        formatted_response: Dict[str, Any],
        question: str,
        raw_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Add contextual enhancements to the response"""

        # Add follow-up suggestions based on content
        if formatted_response["type"] == "analysis":
            formatted_response["follow_up_questions"] = await self._generate_follow_ups(
                question, raw_result
            )

        # Add explanation for complex data
        if raw_result.get("result_count", 0) > 20:
            formatted_response["data_note"] = (
                "This is a large dataset. Would you like me to filter it or focus on specific aspects?"
            )

        # Add time context for date-related queries
        if any(
            word in question.lower()
            for word in ["2024", "2025", "this year", "last year", "quarter"]
        ):
            formatted_response["time_context"] = (
                "I'm showing data for the time period you specified. Let me know if you'd like to see different dates."
            )

        return formatted_response

    async def _generate_follow_ups(
        self, question: str, raw_result: Dict[str, Any]
    ) -> List[str]:
        """Generate intelligent follow-up questions"""

        result_count = raw_result.get("result_count", 0)

        prompt = f"""
        User asked: "{question}"
        Found: {result_count} records
        
        Generate 3 natural follow-up questions they might want to ask next:
        - Build on their current question
        - Dig deeper into the data
        - Explore related aspects
        - Use conversational language
        
        Return only the questions, one per line.
        """

        try:
            follow_ups_text = await self.ai_services.ask_intelligent_llm_async(prompt)
            follow_ups = [q.strip() for q in follow_ups_text.split("\n") if q.strip()]
            return follow_ups[:3]
        except:
            return [
                "Would you like to see this data in a different time period?",
                "What specific aspects would you like me to analyze further?",
                "Should I create a visualization of this data?",
            ]
