#!/usr/bin/env python3
"""
Startup script for HyperGraph Test Configuration Web UI

This script starts the Flask web interface for managing test configurations
and runs. It handles dependency installation and provides instructions for
remote access over SSH.

Author: Quanty 7
"""

import os
import sys
import subprocess
import socket

def check_dependencies():
    """Check if required dependencies are installed."""
    # Map package names to import names
    required_packages = {
        'flask': 'flask',
        'psutil': 'psutil', 
        'pyyaml': 'yaml'
    }
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("Missing required packages:", ', '.join(missing_packages))
        print("\nInstall with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def get_local_ip():
    """Get the local IP address."""
    try:
        # Connect to a remote address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "localhost"

def main():
    print("🧠 HyperGraph Test Configuration Web UI")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('test_hierarchical.py'):
        print("❌ Error: Please run this script from the HyperGraph project root directory")
        print("   (where test_hierarchical.py is located)")
        sys.exit(1)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Missing dependencies. Please install them and try again.")
        sys.exit(1)
    
    print("✅ Dependencies OK")
    
    # Create necessary directories
    print("📁 Creating directories...")
    dirs_to_create = [
        'web_ui/configs',
        'web_ui/runs', 
        'web_ui/config_templates',
        'web_ui/static/css',
        'web_ui/static/js'
    ]
    
    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Directories created")
    
    # Get network information
    local_ip = get_local_ip()
    port = int(os.environ.get('PORT', 5000))
    
    print("\n🚀 Starting Web UI...")
    print(f"   Local URL:  http://localhost:{port}")
    print(f"   Network URL: http://{local_ip}:{port}")
    
    print("\n🔗 SSH Tunnel Setup:")
    print("   For automated SSH tunnel setup, run:")
    print("   python setup_ssh_tunnel.py --create-script")
    print(f"   Then copy the generated script to your local machine")
    print(f"   Manual tunnel: ssh -L {port}:localhost:{port} your_username@{local_ip}")
    
    print("\n📖 Features:")
    print("   • Create and save test configurations")
    print("   • Start test runs remotely")
    print("   • Monitor test progress and logs")
    print("   • View and compare results")
    print("   • Configuration templates")
    
    print("\n" + "=" * 50)
    print("Starting Flask application...\n")
    
    # Change to the web_ui directory and start the app (pass PORT to subprocess)
    try:
        os.chdir('web_ui')
        env = os.environ.copy()
        env['PORT'] = str(port)
        subprocess.run([sys.executable, 'app.py'], env=env)
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down Web UI...")
    except Exception as e:
        print(f"\n❌ Error starting web UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 