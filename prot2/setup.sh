#!/bin/bash

# Behavioral Biometrics Authentication Agent - Setup Script
# Automates the complete project setup and installation process

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="Behavioral Biometrics Authentication Agent"
PROJECT_VERSION="2.1.0"
PYTHON_MIN_VERSION="3.10"
NODE_MIN_VERSION="16"

# Functions
print_banner() {
    echo -e "${PURPLE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                                                              ║"
    echo "║        🔐 Behavioral Biometrics Authentication Agent         ║"
    echo "║                                                              ║"
    echo "║                    Setup & Installation                      ║"
    echo "║                      v${PROJECT_VERSION} - 2024.08.20                        ║"
    echo "║                                                              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "${BLUE}🔹 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Version comparison function
version_ge() {
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

# Check system requirements
check_system() {
    print_step "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="Linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macOS"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        OS="Windows"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    print_success "Operating System: $OS"
    
    # Check Python
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
        if version_ge "$PYTHON_VERSION" "$PYTHON_MIN_VERSION"; then
            print_success "Python $PYTHON_VERSION (>= $PYTHON_MIN_VERSION required)"
        else
            print_error "Python $PYTHON_MIN_VERSION or higher required. Found: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python $PYTHON_MIN_VERSION or higher."
        exit 1
    fi
    
    # Check pip
    if command_exists pip3; then
        print_success "pip3 found"
    elif command_exists pip; then
        print_success "pip found"
    else
        print_error "pip not found. Please install pip."
        exit 1
    fi
    
    # Check git
    if command_exists git; then
        print_success "Git found"
    else
        print_warning "Git not found. Some features may not work."
    fi
    
    # Check curl
    if command_exists curl; then
        print_success "curl found"
    else
        print_warning "curl not found. Installing..."
        install_curl
    fi
}

# Install curl based on OS
install_curl() {
    case $OS in
        "Linux")
            if command_exists apt-get; then
                sudo apt-get update && sudo apt-get install -y curl
            elif command_exists yum; then
                sudo yum install -y curl
            elif command_exists dnf; then
                sudo dnf install -y curl
            else
                print_error "Cannot install curl automatically. Please install manually."
                exit 1
            fi
            ;;
        "macOS")
            if command_exists brew; then
                brew install curl
            else
                print_error "Homebrew not found. Please install curl manually."
                exit 1
            fi
            ;;
        "Windows")
            print_info "Please install curl manually from https://curl.se/windows/"
            ;;
    esac
}

# Setup virtual environment
setup_virtualenv() {
    print_step "Setting up Python virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment already exists. Removing..."
        rm -rf venv
    fi
    
    python3 -m venv venv
    
    # Activate virtual environment
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        print_success "Virtual environment activated"
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
        print_success "Virtual environment activated"
    else
        print_error "Failed to activate virtual environment"
        exit 1
    fi
    
    # Upgrade pip
    pip install --upgrade pip
    print_success "pip upgraded"
}

# Install Python dependencies
install_dependencies() {
    print_step "Installing Python dependencies..."
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found"
        exit 1
    fi
    
    pip install -r requirements.txt
    print_success "Python dependencies installed"
}

# Setup environment configuration
setup_environment() {
    print_step "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        print_info "Creating .env file..."
        
        # Generate random secrets
        SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
        JWT_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
        
        cat > .env << EOF
# Behavioral Biometrics Authentication Agent Configuration
# Generated on $(date)

# Security Keys (IMPORTANT: Change these in production!)
SECRET_KEY=${SECRET_KEY}
JWT_SECRET_KEY=${JWT_SECRET_KEY}

# Application Settings
FLASK_ENV=development
FLASK_DEBUG=True

# Database Configuration
DATABASE_URL=sqlite:///behavioral_auth.db

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

# Development Settings
DEV_PORT=5001
POSTGRES_PASSWORD=secureauth123
REDIS_PASSWORD=secureauth123
EOF
        
        print_success ".env file created with secure random keys"
    else
        print_success ".env file already exists"
    fi
}

# Initialize database
setup_database() {
    print_step "Initializing database..."
    
    python3 -c "
from database.db_manager import DatabaseManager
from config import Config
try:
    db_manager = DatabaseManager(Config.DATABASE_URL)
    db_manager.init_database()
    print('Database initialized successfully')
except Exception as e:
    print(f'Database initialization failed: {e}')
    exit(1)
"
    
    print_success "Database initialized"
}

# Create necessary directories
create_directories() {
    print_step "Creating project directories..."
    
    directories=(
        "logs"
        "data"
        "models"
        "test-reports"
        "monitoring"
        "nginx"
        "static/uploads"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_success "Created directory: $dir"
    done
}

# Setup Docker environment
setup_docker() {
    if command_exists docker && command_exists docker-compose; then
        print_step "Setting up Docker environment..."
        
        # Create docker-compose override for development
        cat > docker-compose.override.yml << EOF
version: '3.8'

services:
  app-dev:
    ports:
      - "5001:5000"
    environment:
      - FLASK_DEBUG=True
    volumes:
      - .:/app
EOF
        
        print_success "Docker environment configured"
        print_info "Run 'docker-compose --profile dev up' for development"
        print_info "Run 'docker-compose up' for production"
    else
        print_warning "Docker not found. Skipping Docker setup."
        print_info "Install Docker to use containerized deployment."
    fi
}

# Run tests
run_tests() {
    print_step "Running test suite..."
    
    if command_exists pytest; then
        pytest --version > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            python3 -m pytest tests/ -v --tb=short || true
            print_success "Test suite completed"
        else
            print_warning "pytest not properly installed. Skipping tests."
        fi
    else
        print_warning "pytest not found. Skipping tests."
    fi
}

# Create startup scripts
create_startup_scripts() {
    print_step "Creating startup scripts..."
    
    # Create start script for Unix systems
    cat > start.sh << 'EOF'
#!/bin/bash
# Start Behavioral Biometrics Authentication Agent

echo "🔐 Starting Behavioral Biometrics Authentication Agent..."

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Start the application
python run.py "$@"
EOF
    
    # Create Windows batch file
    cat > start.bat << 'EOF'
@echo off
echo 🔐 Starting Behavioral Biometrics Authentication Agent...

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Please run setup.sh first.
    exit /b 1
)

REM Start the application
python run.py %*
EOF
    
    # Make scripts executable
    chmod +x start.sh
    chmod +x run.py
    
    print_success "Startup scripts created"
}

# Show completion message
show_completion() {
    echo ""
    echo -e "${GREEN}🎉 Setup completed successfully!${NC}"
    echo ""
    echo -e "${CYAN}Next steps:${NC}"
    echo "1. Activate virtual environment:"
    if [[ "$OS" == "Windows" ]]; then
        echo "   ${YELLOW}venv\\Scripts\\activate${NC}"
    else
        echo "   ${YELLOW}source venv/bin/activate${NC}"
    fi
    echo ""
    echo "2. Start the application:"
    echo "   ${YELLOW}python run.py${NC}"
    echo "   or"
    if [[ "$OS" == "Windows" ]]; then
        echo "   ${YELLOW}start.bat${NC}"
    else
        echo "   ${YELLOW}./start.sh${NC}"
    fi
    echo ""
    echo "3. Open your browser to:"
    echo "   ${YELLOW}http://localhost:5000${NC}"
    echo ""
    echo -e "${CYAN}Additional commands:${NC}"
    echo "  ${YELLOW}python run.py --help${NC}     - Show all options"
    echo "  ${YELLOW}python run.py --test${NC}     - Run test suite"
    echo "  ${YELLOW}python run.py --info${NC}     - Show system information"
    echo ""
    if command_exists docker; then
        echo -e "${CYAN}Docker commands:${NC}"
        echo "  ${YELLOW}docker-compose --profile dev up${NC}   - Development mode"
        echo "  ${YELLOW}docker-compose up${NC}                 - Production mode"
        echo ""
    fi
    echo -e "${GREEN}Happy secure authenticating! 🔐✨${NC}"
}

# Main setup function
main() {
    print_banner
    
    echo -e "${CYAN}This script will set up the $PROJECT_NAME${NC}"
    echo -e "${CYAN}Please ensure you have administrator/sudo privileges if needed.${NC}"
    echo ""
    
    read -p "Continue with setup? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled."
        exit 0
    fi
    
    echo ""
    print_step "Starting setup process..."
    
    # Run setup steps
    check_system
    setup_virtualenv
    install_dependencies
    setup_environment
    create_directories
    setup_database
    create_startup_scripts
    setup_docker
    run_tests
    
    echo ""
    show_completion
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  --check        Only check system requirements"
        echo "  --docker       Setup Docker environment only"
        echo "  --test         Run tests only"
        echo ""
        exit 0
        ;;
    --check)
        print_banner
        check_system
        exit 0
        ;;
    --docker)
        print_banner
        setup_docker
        exit 0
        ;;
    --test)
        print_banner
        run_tests
        exit 0
        ;;
    *)
        main
        ;;
esac