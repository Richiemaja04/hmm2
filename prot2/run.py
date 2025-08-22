#!/usr/bin/env python3
"""
Behavioral Biometrics Authentication Agent - Application Launcher

Enhanced version with intelligent port conflict resolution and better error handling.
"""

import os
import sys
import subprocess
import platform
import logging
import socket
import time
from pathlib import Path
import argparse
import webbrowser
from datetime import datetime

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_colored(message, color=Colors.OKGREEN):
    """Print colored message to terminal."""
    print(f"{color}{message}{Colors.ENDC}")

def print_banner():
    """Print application banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🔐 Behavioral Biometrics Authentication Agent         ║
    ║                                                              ║
    ║              Advanced Continuous Authentication              ║
    ║                     v2.1.0 - 2024.08.20                    ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print_colored(banner, Colors.HEADER)

def check_python_version():
    """Check if Python version meets requirements."""
    print_colored("🔍 Checking Python version...", Colors.OKCYAN)
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print_colored(f"❌ Python 3.10+ required. Current: {version.major}.{version.minor}", Colors.FAIL)
        return False
    
    print_colored(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK", Colors.OKGREEN)
    return True

def is_port_in_use(port, host='localhost'):
    """Check if a port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        return result == 0

def find_available_port(start_port=5000, max_attempts=100):
    """Find an available port starting from start_port."""
    for i in range(max_attempts):
        port = start_port + i
        if not is_port_in_use(port):
            return port
    return None

def kill_process_on_port(port):
    """Kill process running on specified port."""
    try:
        if platform.system() == "Windows":
            # Windows command
            result = subprocess.run(
                f'netstat -ano | findstr :{port}',
                shell=True, capture_output=True, text=True
            )
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5 and f':{port}' in parts[1]:
                        pid = parts[-1]
                        subprocess.run(f'taskkill /F /PID {pid}', shell=True)
                        print_colored(f"✅ Killed process {pid} on port {port}", Colors.OKGREEN)
                        return True
        else:
            # Unix/Linux/macOS command
            result = subprocess.run(
                f'lsof -ti:{port}',
                shell=True, capture_output=True, text=True
            )
            if result.stdout:
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid:
                        subprocess.run(f'kill -9 {pid}', shell=True)
                        print_colored(f"✅ Killed process {pid} on port {port}", Colors.OKGREEN)
                return True
        return False
    except Exception as e:
        print_colored(f"⚠️  Error killing process on port {port}: {e}", Colors.WARNING)
        return False

def get_process_info_on_port(port):
    """Get information about process running on port."""
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                f'netstat -ano | findstr :{port}',
                shell=True, capture_output=True, text=True
            )
            if result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if f':{port}' in line and 'LISTENING' in line:
                        parts = line.split()
                        if len(parts) >= 5:
                            pid = parts[-1]
                            # Get process name
                            proc_result = subprocess.run(
                                f'tasklist /FI "PID eq {pid}" /FO CSV /NH',
                                shell=True, capture_output=True, text=True
                            )
                            if proc_result.stdout:
                                proc_name = proc_result.stdout.split(',')[0].strip('"')
                                return f"PID {pid} ({proc_name})"
        else:
            result = subprocess.run(
                f'lsof -ti:{port}',
                shell=True, capture_output=True, text=True
            )
            if result.stdout:
                pid = result.stdout.strip().split('\n')[0]
                # Get process command
                proc_result = subprocess.run(
                    f'ps -p {pid} -o pid,cmd --no-headers',
                    shell=True, capture_output=True, text=True
                )
                if proc_result.stdout:
                    return proc_result.stdout.strip()
        return None
    except Exception as e:
        return f"Error getting process info: {e}"

def handle_port_conflict(port, auto_resolve=False):
    """Handle port conflicts intelligently."""
    print_colored(f"⚠️  Port {port} is already in use", Colors.WARNING)
    
    # Get process information
    process_info = get_process_info_on_port(port)
    if process_info:
        print_colored(f"   Process: {process_info}", Colors.OKCYAN)
        
        # Check if it's our own application
        if any(keyword in process_info.lower() for keyword in ['python', 'app.py', 'run.py', 'gunicorn', 'flask']):
            print_colored("   🔍 Detected existing instance of this application", Colors.WARNING)
            
            if auto_resolve:
                print_colored("   🔄 Auto-resolving: Killing existing instance...", Colors.OKCYAN)
                if kill_process_on_port(port):
                    time.sleep(2)  # Wait for port to be freed
                    if not is_port_in_use(port):
                        print_colored(f"   ✅ Port {port} is now available", Colors.OKGREEN)
                        return port
            else:
                response = input(f"   Kill existing instance and use port {port}? (y/N): ").lower()
                if response in ['y', 'yes']:
                    if kill_process_on_port(port):
                        time.sleep(2)
                        if not is_port_in_use(port):
                            print_colored(f"   ✅ Port {port} is now available", Colors.OKGREEN)
                            return port
    
    # Find alternative port
    print_colored("   🔍 Searching for alternative port...", Colors.OKCYAN)
    alternative_port = find_available_port(port + 1)
    
    if alternative_port:
        print_colored(f"   ✅ Found alternative port: {alternative_port}", Colors.OKGREEN)
        
        # Update .env.local with new port
        try:
            env_local_path = Path('.env.local')
            env_content = ""
            if env_local_path.exists():
                env_content = env_local_path.read_text()
            
            # Update or add PORT entry
            if 'PORT=' in env_content:
                lines = env_content.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('PORT='):
                        lines[i] = f'PORT={alternative_port}'
                        break
                env_content = '\n'.join(lines)
            else:
                env_content += f'\nPORT={alternative_port}\n'
            
            env_local_path.write_text(env_content)
            print_colored(f"   📝 Updated .env.local with PORT={alternative_port}", Colors.OKGREEN)
            
        except Exception as e:
            print_colored(f"   ⚠️  Could not update .env.local: {e}", Colors.WARNING)
        
        return alternative_port
    else:
        print_colored("   ❌ No alternative ports available", Colors.FAIL)
        return None

def check_dependencies():
    """Check if required dependencies are installed."""
    print_colored("📦 Checking dependencies...", Colors.OKCYAN)
    
    required_packages = [
        'flask',
        'flask_socketio', 
        'sqlalchemy',
        'sklearn',
        'numpy',
        'pandas',
        'jwt'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print_colored(f"  ✅ {package}", Colors.OKGREEN)
        except ImportError:
            print_colored(f"  ❌ {package}", Colors.FAIL)
            missing_packages.append(package)
    
    if missing_packages:
        print_colored(f"\n❌ Missing packages: {', '.join(missing_packages)}", Colors.FAIL)
        print_colored("Run: pip install -r requirements.txt", Colors.WARNING)
        return False
    
    # Check optional TensorFlow
    try:
        import tensorflow as tf
        print_colored(f"  ✅ tensorflow {tf.__version__}", Colors.OKGREEN)
    except ImportError:
        print_colored("  ⚠️  tensorflow (optional) - Neural networks will be disabled", Colors.WARNING)
    
    return True

def check_environment():
    """Check environment configuration."""
    print_colored("🌍 Checking environment...", Colors.OKCYAN)
    
    # Check for .env file
    env_file = Path('.env')
    if not env_file.exists():
        print_colored("⚠️  .env file not found - creating default", Colors.WARNING)
        create_default_env()
    else:
        print_colored("✅ .env file found", Colors.OKGREEN)
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        # Load .env.local for overrides
        env_local = Path('.env.local')
        if env_local.exists():
            load_dotenv('.env.local', override=True)
            print_colored("✅ .env.local loaded (overrides)", Colors.OKGREEN)
            
    except ImportError:
        print_colored("⚠️  python-dotenv not installed - using system environment", Colors.WARNING)
    
    # Check required environment variables
    required_vars = ['SECRET_KEY']
    for var in required_vars:
        if not os.getenv(var):
            print_colored(f"⚠️  {var} not set - using default", Colors.WARNING)
        else:
            print_colored(f"✅ {var} configured", Colors.OKGREEN)
    
    return True

def create_default_env():
    """Create default .env file."""
    import secrets
    
    default_env = f"""# Behavioral Biometrics Authentication Agent Configuration
# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# Security Keys (CHANGE THESE IN PRODUCTION!)
SECRET_KEY={secrets.token_hex(32)}
JWT_SECRET_KEY={secrets.token_hex(32)}

# Application Settings
FLASK_DEBUG=True
FLASK_ENV=development

# Database Configuration
DATABASE_URL=sqlite:///behavioral_auth.db

# Server Configuration
PORT=5000
HOST=0.0.0.0

# Monitoring Configuration
MONITORING_WINDOW_SIZE=30
MIN_CALIBRATION_SAMPLES=50

# Security Thresholds
ANOMALY_THRESHOLD_LOW=0.3
ANOMALY_THRESHOLD_MEDIUM=0.6
ANOMALY_THRESHOLD_HIGH=0.8
ANOMALY_THRESHOLD_CRITICAL=0.9

# Performance Settings
MAX_RESPONSE_TIME=200
MEMORY_THRESHOLD=512
CPU_THRESHOLD=80
"""
    
    with open('.env', 'w') as f:
        f.write(default_env)
    
    print_colored("✅ Created default .env file", Colors.OKGREEN)

def setup_database():
    """Initialize database if needed."""
    print_colored("🗄️  Setting up database...", Colors.OKCYAN)
    
    try:
        from database.db_manager import DatabaseManager
        from config import Config
        
        db_manager = DatabaseManager(Config.DATABASE_URL)
        db_manager.init_database()
        
        print_colored("✅ Database initialized", Colors.OKGREEN)
        return True
        
    except Exception as e:
        print_colored(f"❌ Database setup failed: {e}", Colors.FAIL)
        return False

def run_application(host='localhost', port=5000, debug=True, auto_open=False, auto_resolve_ports=False):
    """Run the Flask application with intelligent port handling."""
    
    # Handle port conflicts
    if is_port_in_use(port, host):
        resolved_port = handle_port_conflict(port, auto_resolve_ports)
        if resolved_port is None:
            print_colored(f"❌ Could not resolve port conflict for {port}", Colors.FAIL)
            return False
        port = resolved_port
    
    print_colored(f"🚀 Starting Behavioral Biometrics Authentication Agent...", Colors.OKGREEN)
    print_colored(f"   Server: http://{host}:{port}", Colors.OKCYAN)
    print_colored(f"   Debug Mode: {'ON' if debug else 'OFF'}", Colors.OKCYAN)
    print_colored(f"   Environment: {os.getenv('FLASK_ENV', 'development')}", Colors.OKCYAN)
    
    if auto_open:
        # Open browser after a short delay
        def open_browser():
            time.sleep(3)
            webbrowser.open(f"http://{host}:{port}")
        
        import threading
        threading.Thread(target=open_browser, daemon=True).start()
        print_colored(f"   🌐 Browser will open automatically", Colors.OKCYAN)
    
    try:
        # Set environment variables
        os.environ['PORT'] = str(port)
        os.environ['HOST'] = host
        
        # Import and run the application
        from app import app, socketio
        
        # Set configuration
        app.config['DEBUG'] = debug
        
        print_colored(f"\n🎯 Application starting on port {port}...", Colors.OKGREEN)
        print_colored(f"   Press Ctrl+C to stop the server", Colors.OKCYAN)
        print_colored(f"   Visit: http://{host}:{port}", Colors.HEADER)
        
        # Run with SocketIO support
        socketio.run(
            app,
            host=host,
            port=port,
            debug=debug,
            use_reloader=debug and not auto_resolve_ports,  # Disable reloader if auto-resolving
            log_output=debug,
            allow_unsafe_werkzeug=True
        )
        
    except KeyboardInterrupt:
        print_colored("\n👋 Application stopped by user", Colors.WARNING)
    except OSError as e:
        if "Address already in use" in str(e):
            print_colored(f"\n❌ Port {port} became unavailable during startup", Colors.FAIL)
            print_colored("   Try running with --auto-resolve to handle port conflicts automatically", Colors.WARNING)
        else:
            print_colored(f"\n❌ Network error: {e}", Colors.FAIL)
        return False
    except Exception as e:
        print_colored(f"\n❌ Application error: {e}", Colors.FAIL)
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """Main entry point with enhanced argument parsing."""
    parser = argparse.ArgumentParser(
        description="Behavioral Biometrics Authentication Agent Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                          # Run with default settings
  python run.py --port 8080              # Run on port 8080
  python run.py --host 0.0.0.0           # Listen on all interfaces
  python run.py --no-debug               # Run without debug mode
  python run.py --auto-resolve           # Auto-resolve port conflicts
  python run.py --check                  # Check system requirements only
  python run.py --kill-port 5000         # Kill process on port 5000
        """
    )
    
    parser.add_argument('--host', default='localhost', 
                       help='Host to bind to (default: localhost)')
    parser.add_argument('--port', type=int, 
                       default=int(os.getenv('PORT', 5000)),
                       help='Port to bind to (default: from PORT env var or 5000)')
    parser.add_argument('--no-debug', action='store_true',
                       help='Disable debug mode')
    parser.add_argument('--no-open', action='store_true',
                       help='Don\'t auto-open browser')
    parser.add_argument('--auto-resolve', action='store_true',
                       help='Automatically resolve port conflicts')
    parser.add_argument('--check', action='store_true',
                       help='Check system requirements only')
    parser.add_argument('--kill-port', type=int, metavar='PORT',
                       help='Kill process running on specified port')
    parser.add_argument('--find-port', type=int, metavar='START', nargs='?', const=5000,
                       help='Find available port starting from number (default: 5000)')
    parser.add_argument('--setup', action='store_true',
                       help='Run initial setup (install deps, setup env, init db)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Set up logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    # Handle special commands first
    if args.kill_port:
        print_colored(f"🔪 Attempting to kill process on port {args.kill_port}...", Colors.OKCYAN)
        process_info = get_process_info_on_port(args.kill_port)
        if process_info:
            print_colored(f"   Found process: {process_info}", Colors.WARNING)
        
        if kill_process_on_port(args.kill_port):
            print_colored(f"✅ Successfully killed process on port {args.kill_port}", Colors.OKGREEN)
        else:
            print_colored(f"❌ No process found on port {args.kill_port} or failed to kill", Colors.FAIL)
        return 0
    
    if args.find_port is not None:
        print_colored(f"🔍 Finding available port starting from {args.find_port}...", Colors.OKCYAN)
        available_port = find_available_port(args.find_port)
        if available_port:
            print_colored(f"✅ Available port found: {available_port}", Colors.OKGREEN)
            print_colored(f"   Start with: python run.py --port {available_port}", Colors.OKCYAN)
        else:
            print_colored(f"❌ No available ports found starting from {args.find_port}", Colors.FAIL)
        return 0
    
    # Handle setup command
    if args.setup:
        print_colored("🔧 Running initial setup...", Colors.HEADER)
        
        # Install dependencies
        print_colored("📦 Installing dependencies...", Colors.OKCYAN)
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                          check=True)
            print_colored("✅ Dependencies installed", Colors.OKGREEN)
        except subprocess.CalledProcessError:
            print_colored("❌ Failed to install dependencies", Colors.FAIL)
            return 1
        except FileNotFoundError:
            print_colored("❌ requirements.txt not found", Colors.FAIL)
            return 1
        
        # Setup environment and database
        if not check_environment():
            return 1
        if not setup_database():
            return 1
        
        print_colored("✅ Setup completed successfully!", Colors.OKGREEN)
        print_colored("   You can now run: python run.py", Colors.OKCYAN)
        return 0
    
    # System checks
    print_colored("🔍 Performing system checks...", Colors.HEADER)
    
    checks_passed = True
    checks_passed &= check_python_version()
    checks_passed &= check_dependencies()
    checks_passed &= check_environment()
    checks_passed &= setup_database()
    
    if not checks_passed:
        print_colored("\n❌ System checks failed.", Colors.FAIL)
        print_colored("   Try running: python run.py --setup", Colors.WARNING)
        return 1
    
    # Handle check-only request
    if args.check:
        print_colored("\n✅ All system checks passed!", Colors.OKGREEN)
        print_colored("   System is ready to run the application", Colors.OKCYAN)
        return 0
    
    print_colored("\n✅ All checks passed! Starting application...", Colors.OKGREEN)
    
    # Show port availability before starting
    if is_port_in_use(args.port, args.host):
        print_colored(f"⚠️  Port {args.port} is currently in use", Colors.WARNING)
        if not args.auto_resolve:
            print_colored("   Use --auto-resolve to handle this automatically", Colors.OKCYAN)
    else:
        print_colored(f"✅ Port {args.port} is available", Colors.OKGREEN)
    
    # Run the application
    success = run_application(
        host=args.host,
        port=args.port,
        debug=not args.no_debug,
        auto_open=not args.no_open,
        auto_resolve_ports=args.auto_resolve
    )
    
    return 0 if success else 1

def get_system_info():
    """Display comprehensive system information."""
    print_colored("💻 System Information:", Colors.OKCYAN)
    
    info = {
        "OS": f"{platform.system()} {platform.release()}",
        "Architecture": platform.machine(),
        "Python": f"{sys.version.split()[0]}",
        "CPU Cores": os.cpu_count(),
        "Working Directory": os.getcwd(),
        "Python Executable": sys.executable
    }
    
    for key, value in info.items():
        print_colored(f"  {key}: {value}", Colors.OKGREEN)
    
    # Show environment variables
    print_colored("\n🌍 Environment Variables:", Colors.OKCYAN)
    env_vars = ['PORT', 'HOST', 'FLASK_ENV', 'FLASK_DEBUG', 'DATABASE_URL']
    for var in env_vars:
        value = os.getenv(var, 'Not set')
        print_colored(f"  {var}: {value}", Colors.OKGREEN)
    
    # Show port status
    print_colored("\n🔌 Port Status:", Colors.OKCYAN)
    common_ports = [5000, 5001, 5002, 8000, 8080, 3000]
    for port in common_ports:
        status = "In use" if is_port_in_use(port) else "Available"
        color = Colors.WARNING if status == "In use" else Colors.OKGREEN
        print_colored(f"  Port {port}: {status}", color)
        
        if status == "In use":
            process_info = get_process_info_on_port(port)
            if process_info:
                print_colored(f"    Process: {process_info}", Colors.OKCYAN)

def install_dependencies_interactive():
    """Interactive dependency installation."""
    print_colored("📦 Interactive Dependency Installation", Colors.HEADER)
    
    try:
        # Check if requirements.txt exists
        if not Path('requirements.txt').exists():
            print_colored("❌ requirements.txt not found", Colors.FAIL)
            return False
        
        # Show what will be installed
        with open('requirements.txt', 'r') as f:
            requirements = f.read().strip().split('\n')
        
        print_colored("The following packages will be installed:", Colors.OKCYAN)
        for req in requirements[:10]:  # Show first 10
            if req.strip() and not req.strip().startswith('#'):
                print_colored(f"  • {req.strip()}", Colors.OKGREEN)
        
        if len(requirements) > 10:
            print_colored(f"  ... and {len(requirements) - 10} more packages", Colors.OKCYAN)
        
        response = input(f"\nProceed with installation? (Y/n): ").lower()
        if response in ['', 'y', 'yes']:
            print_colored("\n📦 Installing packages...", Colors.OKCYAN)
            
            # Install with progress indication
            process = subprocess.Popen(
                [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            
            for line in process.stdout:
                if 'Installing' in line or 'Successfully installed' in line:
                    print_colored(f"  {line.strip()}", Colors.OKGREEN)
            
            process.wait()
            
            if process.returncode == 0:
                print_colored("✅ All dependencies installed successfully!", Colors.OKGREEN)
                return True
            else:
                print_colored("❌ Installation failed", Colors.FAIL)
                return False
        else:
            print_colored("Installation cancelled", Colors.WARNING)
            return False
            
    except Exception as e:
        print_colored(f"❌ Error during installation: {e}", Colors.FAIL)
        return False

if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print_colored("\n👋 Setup interrupted by user", Colors.WARNING)
        sys.exit(1)
    except Exception as e:
        print_colored(f"\n💥 Unexpected error: {e}", Colors.FAIL)
        import traceback
        if '--verbose' in sys.argv or '-v' in sys.argv:
            traceback.print_exc()
        sys.exit(1)