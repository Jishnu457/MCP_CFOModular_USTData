"""
Email service using Microsoft Graph API
"""

import base64
import structlog

logger = structlog.get_logger()


class EmailService:
    """Email service using Microsoft Graph API"""

    def __init__(self, graph_client):
        self.graph_client = graph_client

    async def send_email_with_report(
        self, recipients, subject, body, report_data, report_filename, report_type="pdf"
    ):
        """Send email with report attachment"""
        if not self.graph_client:
            logger.error("Graph client not available")
            return False

        try:
            report_base64 = base64.b64encode(report_data).decode("utf-8")

            message = {
                "subject": subject,
                "body": {"contentType": "HTML", "content": body},
                "toRecipients": [
                    {"emailAddress": {"address": email}}
                    for email in recipients
                    if email
                ],
                "attachments": [
                    {
                        "@odata.type": "#microsoft.graph.fileAttachment",
                        "name": report_filename,
                        "contentBytes": report_base64,
                    }
                ],
            }

            await self.graph_client.me.send_mail.post({"message": message})
            logger.info(
                "Email sent successfully",
                recipients=recipients,
                filename=report_filename,
            )
            return True

        except Exception as e:
            logger.error("Email sending failed", error=str(e))
            return False

    async def send_notification_email(self, recipients, subject, body):
        """Send simple notification email without attachments"""
        if not self.graph_client:
            logger.error("Graph client not available")
            return False

        try:
            message = {
                "subject": subject,
                "body": {"contentType": "HTML", "content": body},
                "toRecipients": [
                    {"emailAddress": {"address": email}}
                    for email in recipients
                    if email
                ],
            }

            await self.graph_client.me.send_mail.post({"message": message})
            logger.info("Notification email sent", recipients=recipients)
            return True

        except Exception as e:
            logger.error("Notification email failed", error=str(e))
            return False
