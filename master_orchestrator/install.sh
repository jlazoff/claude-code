#!/bin/bash

# Master Orchestrator Installation Script
# Automated setup for enterprise-scale agentic orchestration

set -e  # Exit on any error

echo "🚀 Master Orchestrator Installation"
echo "====================================="
echo

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check system requirements
check_requirements() {
    print_info "Checking system requirements..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3.11+ is required but not installed"
    fi
    
    python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    if [[ $(echo "$python_version < 3.11" | bc -l) -eq 1 ]]; then
        print_error "Python 3.11+ is required, found: $python_version"
    fi
    print_status "Python $python_version detected"
    
    # Check if running on macOS (required for hardware integration)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        print_status "macOS detected - hardware integration available"
    else
        print_warning "Non-macOS system - some hardware features may be limited"
    fi
    
    # Check available memory
    if [[ "$OSTYPE" == "darwin"* ]]; then
        total_mem=$(sysctl -n hw.memsize)
        total_mem_gb=$((total_mem / 1024 / 1024 / 1024))
        print_info "Available RAM: ${total_mem_gb}GB"
        
        if [[ $total_mem_gb -lt 8 ]]; then
            print_warning "8GB+ RAM recommended for optimal performance"
        fi
    fi
}

# Install system dependencies
install_system_deps() {
    print_info "Installing system dependencies..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - check for Homebrew
        if ! command -v brew &> /dev/null; then
            print_info "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        # Install required tools
        print_info "Installing required tools via Homebrew..."
        brew install git curl wget docker kubernetes-cli terraform ansible
        
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux - detect package manager
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y git curl wget docker.io kubectl terraform ansible
        elif command -v yum &> /dev/null; then
            sudo yum install -y git curl wget docker kubectl terraform ansible
        else
            print_warning "Unsupported package manager - please install dependencies manually"
        fi
    fi
    
    print_status "System dependencies installed"
}

# Setup Python environment
setup_python_env() {
    print_info "Setting up Python environment..."
    
    # Install uv if not present
    if ! command -v uv &> /dev/null; then
        print_info "Installing uv package manager..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source ~/.bashrc 2>/dev/null || source ~/.zshrc 2>/dev/null || true
    fi
    
    # Create virtual environment
    print_info "Creating virtual environment..."
    uv venv master-orchestrator-env
    source master-orchestrator-env/bin/activate
    
    # Install Python dependencies
    print_info "Installing Python dependencies..."
    uv pip install -e .
    
    print_status "Python environment configured"
}

# Setup Docker environment
setup_docker() {
    print_info "Setting up Docker environment..."
    
    # Start Docker if not running
    if ! docker info &> /dev/null; then
        print_info "Starting Docker..."
        if [[ "$OSTYPE" == "darwin"* ]]; then
            open -a Docker
            print_info "Please start Docker Desktop and run this script again"
            exit 0
        else
            sudo systemctl start docker
            sudo systemctl enable docker
        fi
    fi
    
    # Build custom images if needed
    print_info "Building Docker images..."
    
    # ArangoDB with custom config
    cat > Dockerfile.arangodb << 'EOF'
FROM arangodb/arangodb:latest

# Add custom configuration
COPY arangodb.conf /etc/arangodb3/

# Expose ports
EXPOSE 8529

# Set environment variables
ENV ARANGO_ROOT_PASSWORD=orchestrator123
ENV ARANGO_NO_AUTH=0
EOF
    
    # Create ArangoDB config
    cat > arangodb.conf << 'EOF'
[database]
auto-upgrade = true

[server]
authentication = true
endpoint = tcp://0.0.0.0:8529

[log]
level = info
EOF
    
    docker build -f Dockerfile.arangodb -t master-orchestrator/arangodb .
    
    print_status "Docker environment configured"
}

# Create configuration files
create_configs() {
    print_info "Creating configuration files..."
    
    # Copy example config if config.yaml doesn't exist
    if [ ! -f "config.yaml" ]; then
        cp config.example.yaml config.yaml
        print_info "Created config.yaml from example"
        print_warning "Please edit config.yaml with your specific settings"
    fi
    
    # Create directories
    mkdir -p data logs infrastructure/terraform infrastructure/ansible
    
    # Create basic Terraform configuration
    cat > infrastructure/terraform/main.tf << 'EOF'
# Master Orchestrator Infrastructure

terraform {
  required_version = ">= 1.0"
  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

provider "docker" {
  host = "unix:///var/run/docker.sock"
}

# ArangoDB container
resource "docker_container" "arangodb" {
  image = "master-orchestrator/arangodb:latest"
  name  = "master-orchestrator-arangodb"
  
  ports {
    internal = 8529
    external = 8529
  }
  
  env = [
    "ARANGO_ROOT_PASSWORD=orchestrator123"
  ]
  
  volumes {
    host_path      = "${path.cwd}/data/arangodb"
    container_path = "/var/lib/arangodb3"
  }
}

# Monitoring stack
resource "docker_container" "prometheus" {
  image = "prom/prometheus:latest"
  name  = "master-orchestrator-prometheus"
  
  ports {
    internal = 9090
    external = 9090
  }
}

resource "docker_container" "grafana" {
  image = "grafana/grafana:latest"
  name  = "master-orchestrator-grafana"
  
  ports {
    internal = 3000
    external = 3000
  }
  
  env = [
    "GF_SECURITY_ADMIN_PASSWORD=orchestrator123"
  ]
}
EOF
    
    # Create basic Ansible playbook
    cat > infrastructure/ansible/deploy.yml << 'EOF'
---
- name: Deploy Master Orchestrator Infrastructure
  hosts: localhost
  connection: local
  
  tasks:
    - name: Create data directories
      file:
        path: "{{ item }}"
        state: directory
        mode: '0755'
      loop:
        - ./data/arangodb
        - ./data/prometheus
        - ./data/grafana
        - ./logs
    
    - name: Deploy infrastructure with Terraform
      community.general.terraform:
        project_path: ../terraform
        state: present
        force_init: true
EOF
    
    print_status "Configuration files created"
}

# Setup networking for hardware integration
setup_networking() {
    print_info "Setting up networking for hardware integration..."
    
    # Create network configuration script
    cat > setup_network.sh << 'EOF'
#!/bin/bash

# Master Orchestrator Network Setup
# Configures networking for Mac Studios, Mac Minis, and NAS systems

echo "🌐 Configuring Master Orchestrator Network"

# Enable SSH key-based authentication for remote nodes
if [ ! -f ~/.ssh/id_rsa ]; then
    echo "Generating SSH key for remote access..."
    ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
fi

# Create hosts file entries
echo "
# Master Orchestrator Hardware Nodes
192.168.1.10    mac-studio-1
192.168.1.11    mac-studio-2
192.168.1.20    mac-mini-1
192.168.1.21    mac-mini-2
192.168.1.30    synology-nas
192.168.1.31    asustor-nas
" >> /etc/hosts

echo "✅ Network configuration complete"
echo "📝 Next steps:"
echo "   1. Copy SSH public key to remote nodes:"
echo "      ssh-copy-id user@mac-studio-1"
echo "   2. Update config.yaml with actual IP addresses"
echo "   3. Test connectivity: ping mac-studio-1"
EOF
    
    chmod +x setup_network.sh
    
    print_status "Networking configuration created"
}

# Create startup scripts
create_startup_scripts() {
    print_info "Creating startup scripts..."
    
    # Quick start script
    cat > quick_start.sh << 'EOF'
#!/bin/bash

# Master Orchestrator Quick Start
# Launches the system with all components

echo "🚀 Master Orchestrator Quick Start"
echo "=================================="
echo

# Check if config exists
if [ ! -f "config.yaml" ]; then
    echo "❌ config.yaml not found. Please run install.sh first."
    exit 1
fi

# Activate virtual environment
source master-orchestrator-env/bin/activate

# Start infrastructure
echo "🏗️  Starting infrastructure..."
cd infrastructure/terraform && terraform apply -auto-approve && cd ../..

# Start monitoring
echo "📊 Starting monitoring stack..."
docker-compose up -d prometheus grafana

# Start Master Orchestrator
echo "🤖 Starting Master Orchestrator..."
python start.py --quick-start

echo "✅ Master Orchestrator is running!"
echo "🌐 Web Dashboard: http://localhost:8000"
echo "📊 Monitoring: http://localhost:3000"
echo "🛑 Press Ctrl+C to stop"
EOF
    
    chmod +x quick_start.sh
    
    # Development start script
    cat > dev_start.sh << 'EOF'
#!/bin/bash

# Development startup script
source master-orchestrator-env/bin/activate
export ORCHESTRATOR_ENVIRONMENT=development
export LOG_LEVEL=DEBUG
python start.py --config config.yaml
EOF
    
    chmod +x dev_start.sh
    
    # Production start script
    cat > prod_start.sh << 'EOF'
#!/bin/bash

# Production startup script
source master-orchestrator-env/bin/activate
export ORCHESTRATOR_ENVIRONMENT=production
export LOG_LEVEL=INFO
python start.py --config config.yaml --daemon
EOF
    
    chmod +x prod_start.sh
    
    print_status "Startup scripts created"
}

# Create systemd service for Linux
create_service() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        print_info "Creating systemd service..."
        
        cat > master-orchestrator.service << 'EOF'
[Unit]
Description=Master Orchestrator - Agentic Multi-Project Orchestration System
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=orchestrator
WorkingDirectory=/opt/master-orchestrator
Environment=ORCHESTRATOR_ENVIRONMENT=production
ExecStart=/opt/master-orchestrator/master-orchestrator-env/bin/python start.py --config config.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
        
        print_info "Service file created: master-orchestrator.service"
        print_info "To install: sudo cp master-orchestrator.service /etc/systemd/system/"
        print_info "To enable: sudo systemctl enable master-orchestrator"
    fi
}

# Main installation function
main() {
    echo "🚀 Starting Master Orchestrator Installation"
    echo "============================================="
    echo
    
    # Check if we're in the right directory
    if [ ! -f "pyproject.toml" ]; then
        print_error "Please run this script from the master_orchestrator directory"
    fi
    
    check_requirements
    install_system_deps
    setup_python_env
    setup_docker
    create_configs
    setup_networking
    create_startup_scripts
    create_service
    
    echo
    echo "🎉 Installation Complete!"
    echo "========================="
    echo
    print_status "Master Orchestrator is ready for deployment"
    echo
    print_info "Next steps:"
    echo "  1. Edit config.yaml with your settings"
    echo "  2. Set environment variables for API keys:"
    echo "     export OPENAI_API_KEY=your-key"
    echo "     export ANTHROPIC_API_KEY=your-key"
    echo "     export GOOGLE_API_KEY=your-key"
    echo "  3. Run quick start: ./quick_start.sh"
    echo "  4. Access dashboard: http://localhost:8000"
    echo
    print_info "For hardware integration:"
    echo "  1. Run: ./setup_network.sh"
    echo "  2. Configure SSH access to remote nodes"
    echo "  3. Update IP addresses in config.yaml"
    echo
    print_info "Documentation: See README.md for detailed instructions"
    echo
}

# Run main installation
main "$@"