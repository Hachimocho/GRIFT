#!/usr/bin/env python3
"""
Automated SSH Tunnel Setup for HyperGraph Web UI

This script automates the creation of SSH tunnels for remote access to the
HyperGraph web interface. It can be run on either the local or remote machine
to establish secure tunnels.

Author: Quanty 7
"""

import os
import sys
import subprocess
import socket
import time
import signal
import argparse
from pathlib import Path

class SSHTunnelManager:
    def __init__(self, local_port=5000, remote_port=5000):
        self.local_port = local_port
        self.remote_port = remote_port
        self.tunnel_process = None
        
    def get_server_info(self):
        """Get server connection information."""
        try:
            # Try to detect current server info
            hostname = socket.gethostname()
            local_ip = self.get_local_ip()
            
            # Check if we're on the RIT server based on hostname
            if 'rit.edu' in hostname or 'nsf-gpu' in hostname:
                server_host = hostname
            else:
                server_host = local_ip
            
            return {
                'hostname': hostname,
                'ip': local_ip,
                'server_host': server_host
            }
        except Exception as e:
            print(f"Error getting server info: {e}")
            return None
    
    def get_local_ip(self):
        """Get the local IP address."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except:
            return "localhost"
    
    def is_port_in_use(self, port):
        """Check if a port is already in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return False
            except OSError:
                return True
    
    def find_available_port(self, start_port=5000):
        """Find an available port starting from start_port."""
        port = start_port
        while self.is_port_in_use(port) and port < start_port + 100:
            port += 1
        return port if port < start_port + 100 else None
    
    def create_tunnel_script(self, server_host, username=None):
        """Create tunnel scripts for both Linux/macOS and Windows."""
        if username is None:
            username = input("Enter your username for the server: ").strip()
        
        # Find available local port
        available_port = self.find_available_port(self.local_port)
        if available_port is None:
            print(f"❌ No available ports found starting from {self.local_port}")
            return None
        
        if available_port != self.local_port:
            print(f"⚠️  Port {self.local_port} in use, using {available_port} instead")
            self.local_port = available_port
        
        # Create bash script for Linux/macOS
        bash_script_content = f'''#!/bin/bash
# HyperGraph Web UI SSH Tunnel Script (Linux/macOS)
# Generated automatically by setup_ssh_tunnel.py

echo "🔗 Setting up SSH tunnel for HyperGraph Web UI..."
echo "   Local port:  {self.local_port}"
echo "   Remote port: {self.remote_port}"
echo "   Server:      {username}@{server_host}"
echo ""

# Kill any existing tunnels on this port
pkill -f "ssh.*-L.*{self.local_port}:localhost:{self.remote_port}" 2>/dev/null

echo "🚀 Starting SSH tunnel..."
echo "   Keep this terminal open while using the web UI"
echo "   Press Ctrl+C to close the tunnel"
echo ""

# Create SSH tunnel with compression and keep-alive
ssh -N -C -L {self.local_port}:localhost:{self.remote_port} \\
    -o ServerAliveInterval=60 \\
    -o ServerAliveCountMax=3 \\
    -o ExitOnForwardFailure=yes \\
    {username}@{server_host}
'''
        
        # Create PowerShell script for Windows
        powershell_script_content = f'''# HyperGraph Web UI SSH Tunnel Script (Windows PowerShell)
# Generated automatically by setup_ssh_tunnel.py

Write-Host "[SETUP] Setting up SSH tunnel for HyperGraph Web UI..." -ForegroundColor Green
Write-Host "   Local port:  {self.local_port} (will auto-adjust if busy)" -ForegroundColor Yellow
Write-Host "   Remote port: {self.remote_port}" -ForegroundColor Yellow
Write-Host "   Server:      {username}@{server_host}" -ForegroundColor Yellow
Write-Host ""

# Kill any existing SSH processes on this port
Write-Host "[CHECK] Checking for existing SSH tunnels..." -ForegroundColor Blue
Get-Process ssh -ErrorAction SilentlyContinue | Where-Object {{
    $_.CommandLine -like "*-L*{self.local_port}:localhost:{self.remote_port}*"
}} | Stop-Process -Force -ErrorAction SilentlyContinue

# Function to test if port is available
function Test-Port {{
    param([int]$Port)
    try {{
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
        $listener.Start()
        $listener.Stop()
        return $true
    }} catch {{
        return $false
    }}
}}

# Find available port starting from requested port
$originalPort = {self.local_port}
$testPort = $originalPort
$maxAttempts = 50

Write-Host "[PORT] Checking port availability..." -ForegroundColor Blue
while (-not (Test-Port $testPort) -and ($testPort - $originalPort) -lt $maxAttempts) {{
    $testPort++
}}

if (($testPort - $originalPort) -ge $maxAttempts) {{
    Write-Host "[ERROR] Could not find available port in range $originalPort-$($originalPort + $maxAttempts)" -ForegroundColor Red
    Write-Host "Try closing other applications or restarting your computer" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}}

if ($testPort -ne $originalPort) {{
    Write-Host "[INFO] Port $originalPort is busy, using port $testPort instead" -ForegroundColor Yellow
}}

Write-Host "[START] Starting SSH tunnel..." -ForegroundColor Green
Write-Host "   Keep this PowerShell window open while using the web UI" -ForegroundColor Cyan
Write-Host "   Press Ctrl+C to close the tunnel" -ForegroundColor Cyan
Write-Host "   Access the UI at: http://localhost:$testPort" -ForegroundColor Magenta
Write-Host ""

# Create SSH tunnel with compression and keep-alive
ssh -N -C -L "$testPort`:localhost:{self.remote_port}" `
    -o ServerAliveInterval=60 `
    -o ServerAliveCountMax=3 `
    -o ExitOnForwardFailure=yes `
    {username}@{server_host}
'''
        
        # Create batch file for Windows (fallback)
        batch_script_content = f'''@echo off
REM HyperGraph Web UI SSH Tunnel Script (Windows Batch)
REM Generated automatically by setup_ssh_tunnel.py

echo [SETUP] Setting up SSH tunnel for HyperGraph Web UI...
echo    Local port:  {self.local_port}
echo    Remote port: {self.remote_port}
echo    Server:      {username}@{server_host}
echo.

echo [START] Starting SSH tunnel...
echo    Keep this command prompt open while using the web UI
echo    Press Ctrl+C to close the tunnel
echo    Access the UI at: http://localhost:{self.local_port}
echo.

REM Create SSH tunnel with compression and keep-alive
ssh -N -C -L {self.local_port}:localhost:{self.remote_port} ^
    -o ServerAliveInterval=60 ^
    -o ServerAliveCountMax=3 ^
    -o ExitOnForwardFailure=yes ^
    {username}@{server_host}
'''
        
        # Save all scripts
        bash_script_path = Path("connect_to_hypergraph_ui.sh")
        powershell_script_path = Path("connect_to_hypergraph_ui.ps1")
        batch_script_path = Path("connect_to_hypergraph_ui.bat")
        
        bash_script_path.write_text(bash_script_content)
        bash_script_path.chmod(0o755)
        
        powershell_script_path.write_text(powershell_script_content, encoding='utf-8')
        batch_script_path.write_text(batch_script_content, encoding='utf-8')
        
        return {
            'bash': bash_script_path,
            'powershell': powershell_script_path,
            'batch': batch_script_path,
            'port': available_port
        }
    
    def start_tunnel_background(self, server_host, username):
        """Start SSH tunnel in the background (for automation)."""
        try:
            # Kill any existing tunnels
            subprocess.run(['pkill', '-f', f'ssh.*-L.*{self.local_port}:localhost:{self.remote_port}'], 
                         capture_output=True)
            
            # Start new tunnel
            cmd = [
                'ssh', '-N', '-C', 
                '-L', f'{self.local_port}:localhost:{self.remote_port}',
                '-o', 'ServerAliveInterval=60',
                '-o', 'ServerAliveCountMax=3', 
                '-o', 'ExitOnForwardFailure=yes',
                f'{username}@{server_host}'
            ]
            
            self.tunnel_process = subprocess.Popen(cmd)
            return True
        except Exception as e:
            print(f"❌ Error starting tunnel: {e}")
            return False
    
    def stop_tunnel(self):
        """Stop the SSH tunnel."""
        if self.tunnel_process:
            self.tunnel_process.terminate()
            self.tunnel_process = None
        
        # Also kill any other tunnels on this port
        subprocess.run(['pkill', '-f', f'ssh.*-L.*{self.local_port}:localhost:{self.remote_port}'], 
                     capture_output=True)

def generate_instructions(server_info, tunnel_manager):
    """Generate comprehensive setup instructions."""
    hostname = server_info['hostname']
    ip = server_info['ip']
    
    instructions = f"""
🔗 SSH Tunnel Setup Instructions for HyperGraph Web UI
{'=' * 60}

🖥️  SERVER SIDE (where you're running this script):
   Hostname: {hostname}
   IP:       {ip}
   
   1. Start the HyperGraph Web UI:
      python start_ui.py
   
   2. The UI will be available on port {tunnel_manager.remote_port}

💻 CLIENT SIDE (your local machine):
   
   Option A - Automated Script:
   1. Download and run the generated script:
      ./connect_to_hypergraph_ui.sh
   
   Option B - Manual Command:
   1. Run this SSH command on your local machine:
      ssh -L {tunnel_manager.local_port}:localhost:{tunnel_manager.remote_port} username@{hostname}
   
   2. Keep the SSH session open
   
   3. Open browser to: http://localhost:{tunnel_manager.local_port}

🚀 ONE-LINER SETUP:
   On your local machine, run:
   ssh -L {tunnel_manager.local_port}:localhost:{tunnel_manager.remote_port} username@{hostname} "cd {os.getcwd()} && python start_ui.py"

⚡ BACKGROUND TUNNEL:
   For a persistent background tunnel:
   ssh -fN -L {tunnel_manager.local_port}:localhost:{tunnel_manager.remote_port} username@{hostname}

🔒 Security Notes:
   • The tunnel encrypts all traffic between your machine and the server
   • Only you can access the web interface through the tunnel
   • Close unused tunnels to free up ports

💡 Troubleshooting:
   • If port {tunnel_manager.local_port} is busy: pkill -f "ssh.*-L.*{tunnel_manager.local_port}"
   • Check tunnel status: ps aux | grep ssh
   • Test connection: curl http://localhost:{tunnel_manager.local_port}
"""
    
    return instructions

def main():
    parser = argparse.ArgumentParser(description="Setup SSH tunnel for HyperGraph Web UI")
    parser.add_argument('--local-port', type=int, default=5000, 
                       help='Local port for tunnel (default: 5000)')
    parser.add_argument('--remote-port', type=int, default=5000,
                       help='Remote port for web UI (default: 5000)')
    parser.add_argument('--username', type=str,
                       help='Username for SSH connection')
    parser.add_argument('--create-script', action='store_true',
                       help='Create connection script for local machine')
    parser.add_argument('--instructions-only', action='store_true',
                       help='Only show setup instructions')
    
    args = parser.parse_args()
    
    print("🔗 HyperGraph Web UI SSH Tunnel Setup")
    print("=" * 40)
    
    tunnel_manager = SSHTunnelManager(args.local_port, args.remote_port)
    server_info = tunnel_manager.get_server_info()
    
    if not server_info:
        print("❌ Could not detect server information")
        sys.exit(1)
    
    print(f"📍 Detected server: {server_info['hostname']} ({server_info['ip']})")
    
    if args.instructions_only:
        print(generate_instructions(server_info, tunnel_manager))
        return
    
    if args.create_script:
        print("\n📝 Creating connection scripts...")
        script_results = tunnel_manager.create_tunnel_script(
            server_info['server_host'], args.username
        )
        
        if script_results:
            port = script_results['port']
            print(f"✅ Created scripts for all platforms:")
            print(f"   🐧 Linux/macOS: {script_results['bash']}")
            print(f"   🪟 PowerShell:  {script_results['powershell']}")
            print(f"   🪟 Batch file:   {script_results['batch']}")
            
            print(f"\n💻 Usage on your local machine:")
            print(f"   Linux/macOS: ./{script_results['bash'].name}")
            print(f"   Windows PS:  .\\{script_results['powershell'].name}")
            print(f"   Windows CMD: .\\{script_results['batch'].name}")
            print(f"🌐 Then open: http://localhost:{port}")
            
            # Show platform-specific instructions
            print(f"\n📋 Platform-Specific Instructions:")
            print(f"   🐧 Linux/macOS:")
            print(f"      chmod +x {script_results['bash'].name}")
            print(f"      ./{script_results['bash'].name}")
            print(f"   🪟 Windows PowerShell:")
            print(f"      Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser")
            print(f"      .\\{script_results['powershell'].name}")
            print(f"   🪟 Windows Command Prompt:")
            print(f"      .\\{script_results['batch'].name}")
            
            # Show one example script content
            print(f"\n📄 PowerShell Script Preview:")
            print("-" * 50)
            print(script_results['powershell'].read_text())
            print("-" * 50)
        else:
            print("❌ Failed to create scripts")
            sys.exit(1)
    else:
        # Show comprehensive instructions
        print(generate_instructions(server_info, tunnel_manager))
    
    print("\n✨ Setup complete! Use the instructions above to connect.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1) 