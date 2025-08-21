"""
SharePoint integration service
"""

import time
import requests
import structlog
from config.settings import ConfigManager

logger = structlog.get_logger()


class SharePointUploader:
    """Handle SharePoint file uploads"""

    def __init__(self):
        self.config = ConfigManager.get_sharepoint_config()
        self.access_token = None

    def get_access_token(self):
        """Get access token for SharePoint"""
        try:
            token_url = f'https://login.microsoftonline.com/{self.config["tenant_id"]}/oauth2/v2.0/token'
            token_post_data = {
                "client_id": self.config["client_id"],
                "client_secret": self.config["client_secret"],
                "grant_type": "client_credentials",
                "scope": self.config["scope"],
            }

            token_request = requests.post(token_url, data=token_post_data)
            if token_request.status_code == 200:
                self.access_token = token_request.json()["access_token"]
                logger.info("SharePoint access token obtained successfully")
                return True
            else:
                logger.error(
                    "Error obtaining SharePoint access token",
                    response=token_request.text,
                )
                return False

        except Exception as e:
            logger.error("Failed to get SharePoint access token", error=str(e))
            return False

    def upload_pdf_to_sharepoint(self, pdf_data: bytes, file_name: str) -> bool:
        """Upload PDF to SharePoint"""
        try:
            if not self.access_token and not self.get_access_token():
                logger.error("Cannot upload to SharePoint: No access token")
                return False

            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/octet-stream",
            }

            # Clean filename
            clean_filename = file_name.replace(" ", "_").replace(":", "-")
            if not clean_filename.endswith(".pdf"):
                clean_filename += ".pdf"

            upload_url = f"https://graph.microsoft.com/v1.0/sites/{self.config['site_id']}/drives/{self.config['document_library_id']}/root:/{clean_filename}:/content"

            # Upload with retry logic
            upload_response = self._upload_with_retry(upload_url, headers, pdf_data)

            if upload_response and (
                upload_response.status_code == 200 or upload_response.status_code == 201
            ):
                logger.info(
                    "PDF uploaded to SharePoint successfully", filename=clean_filename
                )
                return True
            else:
                logger.error(
                    "Failed to upload PDF to SharePoint",
                    status_code=(
                        upload_response.status_code
                        if upload_response
                        else "No response"
                    ),
                    filename=clean_filename,
                )
                return False

        except Exception as e:
            logger.error("SharePoint upload error", error=str(e), filename=file_name)
            return False

    def _upload_with_retry(
        self, upload_url: str, headers: dict, file_content: bytes, max_retries: int = 3
    ):
        """Upload with retry logic"""
        retry_delay = 5  # seconds

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.put(upload_url, headers=headers, data=file_content)

                if response.status_code == 200 or response.status_code == 201:
                    logger.info("File uploaded successfully to SharePoint")
                    return response
                elif attempt < max_retries:
                    logger.warning(
                        f"SharePoint upload attempt {attempt}/{max_retries} failed. Retrying in {retry_delay} seconds...",
                        status_code=response.status_code,
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        "SharePoint upload failed after multiple attempts",
                        status_code=response.status_code,
                        response=response.text,
                    )
                    return response

            except Exception as e:
                logger.error(f"SharePoint upload attempt {attempt} error", error=str(e))
                if attempt == max_retries:
                    return None
                time.sleep(retry_delay)

        return None
