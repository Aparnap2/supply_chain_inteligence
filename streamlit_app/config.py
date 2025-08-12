"""
Configuration settings for the Supply Chain Intelligence Dashboard.
"""

import os
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class DashboardConfig:
    """Dashboard configuration settings."""
    
    # Application settings
    app_title: str = "Supply Chain Intelligence Platform"
    app_icon: str = "🔗"
    layout: str = "wide"
    
    # Data settings
    data_refresh_interval: int = 300  # 5 minutes
    cache_ttl: int = 600  # 10 minutes
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    
    # Risk thresholds
    risk_thresholds: Dict[str, float] = None
    
    # Chart colors
    chart_colors: Dict[str, str] = None
    
    # File paths
    data_directory: str = "data"
    results_directory: str = "results"
    exports_directory: str = "exports"
    
    def __post_init__(self):
        """Initialize default values after dataclass creation."""
        if self.risk_thresholds is None:
            self.risk_thresholds = {
                "low": 30.0,
                "medium": 60.0,
                "high": 80.0,
                "critical": 90.0
            }
        
        if self.chart_colors is None:
            self.chart_colors = {
                "primary": "#1f77b4",
                "secondary": "#ff7f0e",
                "success": "#2ca02c",
                "warning": "#ff7f0e",
                "danger": "#d62728",
                "info": "#17a2b8",
                "light": "#f8f9fa",
                "dark": "#343a40"
            }


# Global configuration instance
config = DashboardConfig()

# Environment-specific overrides
if os.getenv("STREAMLIT_ENV") == "production":
    config.data_refresh_interval = 600  # 10 minutes in production
    config.cache_ttl = 1800  # 30 minutes in production

# Dashboard layout configuration
LAYOUT_CONFIG = {
    "sidebar_width": 300,
    "main_content_padding": "1rem",
    "chart_height": 400,
    "table_height": 500,
    "metric_card_height": 120
}

# Visualization configuration
VIZ_CONFIG = {
    "plotly_theme": "plotly_white",
    "default_chart_height": 400,
    "heatmap_colorscale": "RdYlBu_r",
    "map_style": "open-street-map",
    "animation_duration": 500
}

# Table configuration
TABLE_CONFIG = {
    "page_size": 20,
    "sortable": True,
    "filterable": True,
    "exportable": True,
    "column_widths": {
        "name": 200,
        "country": 120,
        "region": 120,
        "industry": 150,
        "risk_level": 100,
        "risk_score": 100,
        "criticality_score": 120
    }
}

# Export configuration
EXPORT_CONFIG = {
    "formats": ["csv", "json", "pdf"],
    "csv_separator": ",",
    "json_indent": 2,
    "pdf_orientation": "landscape",
    "include_charts": True
}

# Navigation configuration
NAV_CONFIG = {
    "pages": [
        {"name": "Dashboard", "icon": "📊", "path": "/"},
        {"name": "Suppliers", "icon": "🏭", "path": "/suppliers"},
        {"name": "Risk Analysis", "icon": "⚠️", "path": "/risks"},
        {"name": "Reports", "icon": "📋", "path": "/reports"},
        {"name": "Settings", "icon": "⚙️", "path": "/settings"}
    ],
    "show_sidebar": True,
    "collapsible": True
}

# API configuration
API_CONFIG = {
    "base_url": os.getenv("API_BASE_URL", "http://localhost:8000"),
    "timeout": 30,
    "retry_attempts": 3,
    "retry_delay": 1
}

# Security configuration
SECURITY_CONFIG = {
    "enable_authentication": os.getenv("ENABLE_AUTH", "false").lower() == "true",
    "session_timeout": 3600,  # 1 hour
    "max_login_attempts": 3,
    "password_min_length": 8
}

# Logging configuration
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_path": "logs/dashboard.log",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# Feature flags
FEATURE_FLAGS = {
    "enable_real_time_updates": True,
    "enable_advanced_analytics": True,
    "enable_ml_explanations": True,
    "enable_export_scheduling": False,
    "enable_user_preferences": True,
    "enable_dark_mode": True
}

# Performance configuration
PERFORMANCE_CONFIG = {
    "enable_caching": True,
    "cache_size": 100,
    "lazy_loading": True,
    "pagination_size": 50,
    "chart_animation": True
}

# Notification configuration
NOTIFICATION_CONFIG = {
    "enable_notifications": True,
    "notification_types": ["info", "warning", "error", "success"],
    "auto_dismiss_timeout": 5000,  # 5 seconds
    "max_notifications": 5
}

# Help and documentation
HELP_CONFIG = {
    "show_help_tooltips": True,
    "documentation_url": "https://docs.supplychainai.com",
    "support_email": "support@supplychainai.com",
    "version": "1.0.0"
}

def get_config() -> DashboardConfig:
    """Get the global configuration instance."""
    return config

def update_config(**kwargs) -> None:
    """Update configuration settings."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

def reset_config() -> None:
    """Reset configuration to defaults."""
    global config
    config = DashboardConfig()