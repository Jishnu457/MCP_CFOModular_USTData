"""
Session management utilities
"""

import time
from datetime import datetime
from typing import Optional


class SessionManager:
    """Centralized session management"""

    @staticmethod
    def generate_new_session_id():
        """Generate a new unique session ID for the current day"""
        now = datetime.now()
        date_string = now.strftime("%Y%m%d")
        timestamp = int(time.time() * 1000)  # milliseconds for uniqueness
        return f"powerbi_{date_string}_{timestamp}"

    @staticmethod
    def get_session_id_from_request(session: Optional[str] = None):
        """Enhanced session management with multiple sessions per day"""
        if session and (session.startswith("powerbi_") or session == "new"):
            if session == "new":
                # Generate a completely new session
                return SessionManager.generate_new_session_id()
            return session

        # Default fallback
        now = datetime.now()
        date_string = now.strftime("%Y%m%d")
        return f"powerbi_{date_string}_default"
