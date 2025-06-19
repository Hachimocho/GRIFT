# Windows SSH Tunnel Troubleshooting Guide

## Common Port Issues on Windows

### Port 5000 Permission Denied
**Problem**: `bind [127.0.0.1]:5000: Permission denied`

**Cause**: Port 5000 is commonly reserved or used by Windows services, including:
- Windows 10/11 Mobile Hotspot service
- UPnP device services  
- Some Microsoft services
- Other applications

**Solutions**:
1. **Use Updated Scripts** (recommended) - The new PowerShell scripts automatically find available ports
2. **Manual Port Selection** - Use a different port: `ssh -L 5001:localhost:5000 user@server`
3. **Check What's Using Port 5000**:
   ```powershell
   netstat -ano | findstr :5000
   Get-Process -Id [PID_FROM_ABOVE]
   ```

### PowerShell Execution Policy
**Problem**: "Execution policy prevents script from running"

**Solution**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### SSH Not Found
**Problem**: "ssh is not recognized as internal or external command"

**Solutions**:
1. **Windows 10/11**: SSH should be built-in. Enable it:
   ```
   Settings > Apps > Optional Features > Add Feature > OpenSSH Client
   ```
2. **Alternative**: Install Git for Windows (includes SSH)
3. **Check Installation**:
   ```powershell
   ssh -V
   ```

## Automatic Port Detection

The updated PowerShell scripts now include automatic port detection:

1. **Test Original Port** (5000) - Try to bind to check availability
2. **Increment if Busy** - Automatically try 5001, 5002, etc.
3. **Range Limit** - Tests up to 50 ports above the original
4. **Clear Feedback** - Shows which port is actually being used

Example output:
```
[PORT] Checking port availability...
[INFO] Port 5000 is busy, using port 5001 instead
[START] Starting SSH tunnel...
   Access the UI at: http://localhost:5001
```

## Manual Port Selection

If you prefer to specify a port manually:

**PowerShell**:
```powershell
ssh -L 5001:localhost:5000 user@server
```

**Command Prompt**:
```cmd
ssh -L 5001:localhost:5000 user@server
```

Then access the UI at `http://localhost:5001`

## Common Windows-Specific Issues

### 1. Windows Firewall
- May prompt for SSH connection permission
- Allow SSH through firewall when prompted

### 2. Network Adapters
- VPN connections can interfere with local port binding
- Try disconnecting VPN temporarily if issues persist

### 3. Service Conflicts
Common services that use port 5000:
- **Windows Mobile Hotspot** - Can be disabled in Settings
- **UPnP Services** - Part of Windows media/device sharing
- **Development Servers** - Visual Studio, Node.js, etc.

### 4. Unicode/Encoding Issues
- Fixed in latest scripts by replacing emojis with ASCII
- Ensure PowerShell is using UTF-8: `$OutputEncoding = [Console]::OutputEncoding`

## Quick Diagnostics

Run these commands to check your system:

```powershell
# Check if SSH is available
ssh -V

# Check what's using port 5000
netstat -ano | findstr :5000

# Test port availability
Test-NetConnection -ComputerName localhost -Port 5000

# Check PowerShell execution policy
Get-ExecutionPolicy
```

## Alternative Solutions

### 1. Use Different Default Port
Edit the server startup to use a different port:
```bash
python start_ui.py --port 8080
```

### 2. Use Windows Subsystem for Linux (WSL)
If available, WSL provides a Linux environment with better SSH compatibility.

### 3. Use PuTTY for Tunneling
Alternative to built-in SSH:
```
putty -ssh -L 5001:localhost:5000 user@server
```

## Getting Help

If issues persist:
1. **Check the generated scripts** - They include automatic port detection
2. **Try manual port selection** - Use ports 5001, 8080, or 8000
3. **Run diagnostics** - Use the commands above to identify conflicts
4. **Restart services** - Some Windows services can be restarted to free ports

## Success Indicators

When working correctly, you should see:
```
[CONNECT] Connecting to HyperGraph Web UI...
[PORT] Checking port availability...
[START] Starting SSH tunnel...
   Access the UI at: http://localhost:5001
```

Then access `http://localhost:5001` in your browser. 