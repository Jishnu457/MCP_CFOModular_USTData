"""
FastAPI middleware setup
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

from config.settings import AppSettings

def setup_middleware(app: FastAPI):
    """Setup all middleware for the FastAPI app"""
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=AppSettings.CORS_ORIGINS,
        allow_credentials=AppSettings.CORS_CREDENTIALS,
        allow_methods=AppSettings.CORS_METHODS,
        allow_headers=AppSettings.CORS_HEADERS,
        expose_headers=AppSettings.CORS_HEADERS
    )
    
    # Rate limiting
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(429, _rate_limit_exceeded_handler)