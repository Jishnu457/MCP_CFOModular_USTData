"""
Analytics API endpoints
"""
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request, Query
from slowapi import Limiter
from slowapi.util import get_remote_address

from models.requests import IntelligentRequest, ReportRequest
from utils.session_manager import SessionManager
from config.settings import AppSettings

# These will be injected by main.py
analytics_engine = None
ai_services = None
email_service = None
report_generator = None

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

@router.post("/fabric/intelligent")
@limiter.limit(AppSettings.RATE_LIMIT)
async def intelligent_analyze_endpoint(
    req: IntelligentRequest, 
    background_tasks: BackgroundTasks, 
    request: Request,
    session: Optional[str] = Query(None, description="Session ID")
):
    """Enhanced endpoint with AI insights and email notification"""
    if not analytics_engine:
        raise HTTPException(status_code=500, detail="Analytics engine not initialized")
        
    try:
        session_id = SessionManager.get_session_id_from_request(session)
        
        # Process the question with enhanced capabilities
        result = await analytics_engine.cached_intelligent_analyze(
            req.question, 
            session_id, 
            req.enable_ai_insights
        )
        
        if "error" in result and result.get("response_type") != "conversational":
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Send notification email if requested
        if req.enable_email_notification and req.email_recipients and ai_services and ai_services.graph_client:
            async def send_notification():
                try:
                    subject = f"Analytics Result: {req.question[:50]}..."
                    
                    body = f"""
                    <h2>Analytics Notification</h2>
                    <p><strong>Question:</strong> {req.question}</p>
                    <p><strong>Results:</strong> {result.get('result_count', 0)} records found</p>
                    
                    {f"<p><strong>Analysis:</strong></p><p>{result.get('analysis', '')[:500]}...</p>" if result.get('analysis') else ""}
                    
                    {f"<p><strong>AI Insights:</strong></p><p>{result.get('ai_insights', '')[:500]}...</p>" if result.get('ai_insights') else ""}
                    
                    <p>For full details, please check the analytics dashboard.</p>
                    
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    """
                    
                    await email_service.send_notification_email(
                        req.email_recipients, subject, body
                    )
                    
                except Exception as e:
                    pass  # Fail silently for background task
            
            background_tasks.add_task(send_notification)
            result["email_notification_sent"] = True
        
        result["session_id"] = session_id
        result["features_enabled"] = {
            "ai_insights": req.enable_ai_insights and ai_services and ai_services.ai_foundry_enabled,
            "email_notification": req.enable_email_notification and bool(ai_services and ai_services.graph_client),
            "ai_foundry_available": ai_services and ai_services.ai_foundry_enabled,
            "graph_api_available": bool(ai_services and ai_services.graph_client),
            "chat_context": True 
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/intelligent-workflow")
@limiter.limit("5/minute")
async def intelligent_workflow_endpoint(
    req: ReportRequest,
    background_tasks: BackgroundTasks,
    request: Request
):
    """Complete intelligent workflow with proper error logging"""
    
    if not analytics_engine or not report_generator:
        raise HTTPException(status_code=500, detail="Required services not initialized")
    
    try:
        # Generate tracking info
        import uuid
        import structlog
        
        logger = structlog.get_logger()
        workflow_id = str(uuid.uuid4())[:8]
        filename = f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        async def run_workflow():
            try:
                logger.info("🚀 Starting workflow", workflow_id=workflow_id, query=req.data_query)
                
                # Step 1: Analyze data
                analysis_result = await analytics_engine.cached_intelligent_analyze(
                    req.data_query, 
                    enable_ai_insights=req.include_ai_analysis,
                    return_raw_data=True
                )
                
                if "error" in analysis_result:
                    logger.error("❌ Data analysis failed", workflow_id=workflow_id, error=analysis_result.get("error"))
                    return
                
                data = analysis_result.get("sample_data", [])
                logger.info("✅ Data analysis completed", workflow_id=workflow_id, records=len(data))
                
                if not data:
                    logger.warning("⚠️ No data found", workflow_id=workflow_id)
                    return
                
                # Step 2: Prepare analysis text
                if req.include_ai_analysis and "enhanced_analysis" in analysis_result:
                    analysis_text = analysis_result["enhanced_analysis"]
                else:
                    analysis_text = analysis_result.get("analysis", "Analysis not available")
                
                # Step 3: Generate PDF
                logger.info("📄 Starting PDF generation", workflow_id=workflow_id)
                
                if req.report_format == "pdf":
                    try:
                        report_data = await report_generator.generate_pdf_report(
                            data, 
                            analysis_text,
                            f"{req.report_type.title()} Report",
                            req.data_query
                        )
                        
                        if not report_data:
                            logger.error("❌ PDF generation returned no data", workflow_id=workflow_id)
                            return
                        
                        logger.info("✅ PDF generated successfully", workflow_id=workflow_id, size=len(report_data))
                        
                        # Step 4: Upload to SharePoint (instead of direct email)
                        try:
                            from services.sharepoint_service import SharePointUploader
                            
                            logger.info("📤 Starting SharePoint upload", workflow_id=workflow_id, filename=filename)
                            
                            sharepoint_uploader = SharePointUploader()
                            upload_success = sharepoint_uploader.upload_pdf_to_sharepoint(report_data, filename)
                            
                            if upload_success:
                                logger.info("✅ SharePoint upload successful", workflow_id=workflow_id, filename=filename)
                                logger.info("🔄 Power Automate will now handle email delivery", workflow_id=workflow_id)
                            else:
                                logger.error("❌ SharePoint upload failed", workflow_id=workflow_id, filename=filename)
                                return
                                
                        except Exception as sharepoint_error:
                            logger.error("❌ SharePoint upload error", 
                                    workflow_id=workflow_id, 
                                    error=str(sharepoint_error), 
                                    error_type=type(sharepoint_error).__name__)
                            return

                        logger.info("🎉 Workflow completed", workflow_id=workflow_id)
                        
                        '''# Step 4: Send email
                        if req.email_recipients and ai_services and ai_services.graph_client and email_service:
                            try:
                                logger.info("📧 Sending email", workflow_id=workflow_id, recipients=req.email_recipients)
                                
                                await email_service.send_email_with_report(
                                    recipients=req.email_recipients,
                                    subject=req.subject_hint or f"Analytics Report - {datetime.now().strftime('%Y-%m-%d')}",
                                    body=f"<h2>Analytics Report</h2><p>Please find attached the requested {req.report_type} analytics report.</p><p>Query: {req.data_query}</p>",
                                    report_data=report_data,
                                    report_filename=filename,
                                    report_type="pdf"
                                )
                                logger.info("✅ Email sent successfully", workflow_id=workflow_id)
                                
                            except Exception as email_error:
                                logger.error("❌ Email failed", workflow_id=workflow_id, error=str(email_error), error_type=type(email_error).__name__)
                        else:
                            logger.warning("⚠️ Email not configured", 
                                         workflow_id=workflow_id,
                                         has_recipients=bool(req.email_recipients),
                                         has_graph=bool(ai_services and ai_services.graph_client),
                                         has_email_service=bool(email_service))
                        
                        logger.info("🎉 Workflow completed", workflow_id=workflow_id)'''
                        
                    except Exception as pdf_error:
                        logger.error("❌ PDF generation failed", workflow_id=workflow_id, error=str(pdf_error), error_type=type(pdf_error).__name__)
                        import traceback
                        logger.error("PDF error traceback", workflow_id=workflow_id, traceback=traceback.format_exc())
                
            except Exception as workflow_error:
                logger.error("💥 Workflow completely failed", 
                           workflow_id=workflow_id, 
                           error=str(workflow_error), 
                           error_type=type(workflow_error).__name__)
                import traceback
                logger.error("Workflow error traceback", workflow_id=workflow_id, traceback=traceback.format_exc())
        
        background_tasks.add_task(run_workflow)
        
        return {
            "status": "workflow_started",
            "message": "Intelligent workflow initiated. Report will be generated and emailed if configured.",
            "workflow_id": workflow_id,
            "expected_filename": filename,
            "timestamp": datetime.now().isoformat(),
            "debug_info": {
                "has_analytics_engine": bool(analytics_engine),
                "has_report_generator": bool(report_generator),
                "has_email_service": bool(email_service),
                "has_ai_services": bool(ai_services),
                "has_graph_client": bool(ai_services and ai_services.graph_client) if ai_services else False,
                "email_recipients": len(req.email_recipients) if req.email_recipients else 0
            }
        }
        
    except Exception as e:
        logger.error("Failed to start workflow", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/fabric/capabilities")
def get_capabilities():
    return {
        "capabilities": "Natural language query analysis with KQL persistence",
        "example_questions": [
            "What is the average cyber risk score?",
            "Show critical vulnerabilities (CVSS >= 7.0)",
            "Count unpatched devices by type",
            "Show login failure trends over time",
            "What are their departments?"
        ],
        "calculation_features": [
            "SQL-based stats",
            "Aggregations and percentages", 
            "Dynamic risk scores",
            "Trend analysis",
            "Group-based comparisons",
            "Real-time metrics"
        ],
        "intelligence_features": [
            "Natural language understanding",
            "Context-aware answers",
            "Proactive suggestions",
            "Detailed explanations",
            "Business insights"
        ],
        "visualization_features": [
            "Smart chart generation",
            "Bar charts for comparisons",
            "Line charts for trends",
            "Pie charts for distributions",
            "Stacked bars for grouped data"
        ]
    }

'''@router.post("/fabric/intelligent")
@limiter.limit(AppSettings.RATE_LIMIT)
async def intelligent_analyze_endpoint(
    req: IntelligentRequest, 
    background_tasks: BackgroundTasks, 
    request: Request,
    session: Optional[str] = Query(None, description="Session ID")
):
    """Enhanced endpoint with AI insights and email notification"""
    try:
        session_id = SessionManager.get_session_id_from_request(session)
        
        # Process the question with enhanced capabilities
        result = await analytics_engine.cached_intelligent_analyze(
            req.question, 
            session_id, 
            req.enable_ai_insights
        )
        
        if "error" in result and result.get("response_type") != "conversational":
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Send notification email if requested
        if req.enable_email_notification and req.email_recipients and ai_services.graph_client:
            async def send_notification():
                try:
                    subject = f"Analytics Result: {req.question[:50]}..."
                    
                    body = f"""
                    <h2>Analytics Notification</h2>
                    <p><strong>Question:</strong> {req.question}</p>
                    <p><strong>Results:</strong> {result.get('result_count', 0)} records found</p>
                    
                    {f"<p><strong>Analysis:</strong></p><p>{result.get('analysis', '')[:500]}...</p>" if result.get('analysis') else ""}
                    
                    {f"<p><strong>AI Insights:</strong></p><p>{result.get('ai_insights', '')[:500]}...</p>" if result.get('ai_insights') else ""}
                    
                    <p>For full details, please check the analytics dashboard.</p>
                    
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    """
                    
                    await email_service.send_notification_email(
                        req.email_recipients, subject, body
                    )
                    
                except Exception as e:
                    pass  # Fail silently for background task
            
            background_tasks.add_task(send_notification)
            result["email_notification_sent"] = True
        
        result["session_id"] = session_id
        result["features_enabled"] = {
            "ai_insights": req.enable_ai_insights and ai_services.ai_foundry_enabled,
            "email_notification": req.enable_email_notification and bool(ai_services.graph_client),
            "ai_foundry_available": ai_services.ai_foundry_enabled,
            "graph_api_available": bool(ai_services.graph_client),
            "chat_context": True 
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))'''

'''@router.post("/intelligent-workflow")
@limiter.limit("5/minute")
async def intelligent_workflow_endpoint(
    req: ReportRequest,
    background_tasks: BackgroundTasks,
    request: Request
):
    """Complete intelligent workflow - analyze, generate report, and optionally send email"""
    
    try:
        async def run_workflow():
            try:
                # Process the data query
                analysis_result = await analytics_engine.cached_intelligent_analyze(
                    req.data_query, 
                    enable_ai_insights=req.include_ai_analysis
                )
                
                if "error" not in analysis_result:
                    data = analysis_result.get("sample_data", [])
                    
                    # Handle analysis data properly
                    if req.include_ai_analysis and "enhanced_analysis" in analysis_result:
                        analysis_text = analysis_result["enhanced_analysis"]
                    else:
                        analysis_text = analysis_result.get("analysis", "Analysis not available")
                    
                    # Generate report
                    if req.report_format == "pdf":
                        report_data = await report_generator.generate_pdf_report(
                            data, 
                            analysis_text,
                            f"{req.report_type.title()} Report",
                            req.data_query
                        )
                        
                        if report_data:
                            filename = f"analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                            
                            # Send email if configured
                            if req.email_recipients and ai_services.graph_client:
                                try:
                                    await email_service.send_email_with_report(
                                        recipients=req.email_recipients,
                                        subject=req.subject_hint or f"Analytics Report - {datetime.now().strftime('%Y-%m-%d')}",
                                        body=f"<h2>Analytics Report</h2><p>Please find attached the requested {req.report_type} analytics report.</p><p>Query: {req.data_query}</p>",
                                        report_data=report_data,
                                        report_filename=filename,
                                        report_type="pdf"
                                    )
                                except Exception:
                                    pass  # Fail silently for background task
                
            except Exception:
                pass  # Fail silently for background task
        
        background_tasks.add_task(run_workflow)
        
        return {
            "status": "workflow_started",
            "message": "Intelligent workflow initiated. Report will be generated and emailed if configured.",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))'''

'''@router.get("/fabric/capabilities")
def get_capabilities():
    return {
        "capabilities": "Natural language query analysis with KQL persistence",
        "example_questions": [
            "What is the average cyber risk score?",
            "Show critical vulnerabilities (CVSS >= 7.0)",
            "Count unpatched devices by type",
            "Show login failure trends over time",
            "What are their departments?"
        ],
        "calculation_features": [
            "SQL-based stats",
            "Aggregations and percentages", 
            "Dynamic risk scores",
            "Trend analysis",
            "Group-based comparisons",
            "Real-time metrics"
        ],
        "intelligence_features": [
            "Natural language understanding",
            "Context-aware answers",
            "Proactive suggestions",
            "Detailed explanations",
            "Business insights"
        ],
        "visualization_features": [
            "Smart chart generation",
            "Bar charts for comparisons",
            "Line charts for trends",
            "Pie charts for distributions",
            "Stacked bars for grouped data"
        ]
    }'''