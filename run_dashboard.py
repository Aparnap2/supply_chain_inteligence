#!/usr/bin/env python3
"""
Launch script for the Supply Chain Intelligence Dashboard.
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    """Launch the Streamlit dashboard."""
    
    # Get the project root directory
    project_root = Path(__file__).parent
    
    # Set the PYTHONPATH to include the project root
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)
    
    # Path to the main dashboard file
    dashboard_path = project_root / "streamlit_app" / "main.py"
    
    # Check if the dashboard file exists
    if not dashboard_path.exists():
        print(f"Error: Dashboard file not found at {dashboard_path}")
        sys.exit(1)
    
    # Streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false"
    ]
    
    print("🚀 Starting Supply Chain Intelligence Dashboard...")
    print(f"📁 Project root: {project_root}")
    print(f"🌐 Dashboard will be available at: http://localhost:8501")
    print("🔄 Press Ctrl+C to stop the dashboard")
    print("-" * 60)
    
    try:
        # Launch Streamlit
        subprocess.run(cmd, env=env, cwd=project_root)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()