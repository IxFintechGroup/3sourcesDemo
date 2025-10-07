#!/usr/bin/env python3
"""
demo.py — Launcher for the Multi-Source Daily OHLCV Streamlit app
This script provides a simple way to launch the Streamlit application.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit app."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Path to the streamlit app
    app_path = script_dir / "streamlit_app.py"
    
    # Check if the app file exists
    if not app_path.exists():
        print(f"Error: {app_path} not found!")
        print("Please make sure streamlit_app.py exists in the same directory as demo.py")
        sys.exit(1)
    
    # Check if sources_nokey.py exists
    sources_path = script_dir / "sources_nokey.py"
    if not sources_path.exists():
        print(f"Error: {sources_path} not found!")
        print("Please make sure sources_nokey.py exists in the same directory as demo.py")
        sys.exit(1)
    
    print("🚀 Launching Multi-Source Daily OHLCV Streamlit App...")
    print("📊 This app fetches daily OHLCV data from CMC, CryptoCompare, and CoinGecko")
    print("🔗 Opening in your default browser...")
    print()
    
    try:
        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.headless", "false"
        ], cwd=script_dir)
    except KeyboardInterrupt:
        print("\n👋 App closed by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        print("\n💡 Try running manually:")
        print(f"   streamlit run {app_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()