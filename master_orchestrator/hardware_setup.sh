#!/bin/bash

# Master Orchestrator Hardware Setup Script
# Configures Mac Studios, Mac Minis, and NAS systems for distributed operation

set -e

echo "🖥️  Master Orchestrator Hardware Setup"
echo "======================================="
echo

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Configuration
MAC_STUDIOS=("192.168.1.10" "192.168.1.11")
MAC_MINIS=("192.168.1.20" "192.168.1.21") 
NAS_SYSTEMS=("192.168.1.30" "192.168.1.31")
SSH_USER="orchestrator"
SSH_KEY_PATH="$HOME/.ssh/master_orchestrator_rsa"

# Generate SSH key pair for secure communication
setup_ssh_keys() {
    print_info "Setting up SSH keys for secure communication..."
    
    if [ ! -f "$SSH_KEY_PATH" ]; then
        ssh-keygen -t rsa -b 4096 -f "$SSH_KEY_PATH" -N "" -C "master-orchestrator@$(hostname)"
        print_status "Generated SSH key pair"
    else
        print_info "SSH key already exists"
    fi
    
    # Set proper permissions
    chmod 600 "$SSH_KEY_PATH"
    chmod 644 "${SSH_KEY_PATH}.pub"
}

# Configure local machine as control center
setup_control_center() {
    print_info "Setting up local machine as control center..."
    
    # Install required tools for macOS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # Check for Homebrew
        if ! command -v brew &> /dev/null; then
            print_info "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        # Install network tools
        brew install nmap ansible kubectl docker terraform
        
        # Install monitoring tools
        brew install htop iftop
        
    fi
    
    # Create configuration directories
    mkdir -p ~/.config/master-orchestrator/{nodes,monitoring,backups}
    
    print_status "Control center configured"
}

# Test network connectivity to all nodes
test_connectivity() {
    print_info "Testing network connectivity to all nodes..."
    
    local failed_nodes=()
    
    # Test Mac Studios
    for studio in "${MAC_STUDIOS[@]}"; do
        if ping -c 1 -W 2000 "$studio" &> /dev/null; then
            print_status "Mac Studio $studio is reachable"
        else
            print_warning "Mac Studio $studio is not reachable"
            failed_nodes+=("mac-studio-$studio")
        fi
    done
    
    # Test Mac Minis
    for mini in "${MAC_MINIS[@]}"; do
        if ping -c 1 -W 2000 "$mini" &> /dev/null; then
            print_status "Mac Mini $mini is reachable"
        else
            print_warning "Mac Mini $mini is not reachable"
            failed_nodes+=("mac-mini-$mini")
        fi
    done
    
    # Test NAS systems
    for nas in "${NAS_SYSTEMS[@]}"; do
        if ping -c 1 -W 2000 "$nas" &> /dev/null; then
            print_status "NAS $nas is reachable"
        else
            print_warning "NAS $nas is not reachable"
            failed_nodes+=("nas-$nas")
        fi
    done
    
    if [ ${#failed_nodes[@]} -gt 0 ]; then
        print_warning "Some nodes are not reachable: ${failed_nodes[*]}"
        print_info "Please ensure all nodes are powered on and connected to the network"
    else
        print_status "All nodes are reachable"
    fi
}

# Deploy SSH keys to remote nodes
deploy_ssh_keys() {
    print_info "Deploying SSH keys to remote nodes..."
    
    local public_key=$(cat "${SSH_KEY_PATH}.pub")
    
    # Deploy to Mac Studios
    for studio in "${MAC_STUDIOS[@]}"; do
        print_info "Deploying SSH key to Mac Studio $studio..."
        
        # Try to copy SSH key (may require manual intervention)
        if ssh-copy-id -i "$SSH_KEY_PATH" "${SSH_USER}@${studio}" 2>/dev/null; then
            print_status "SSH key deployed to Mac Studio $studio"
        else
            print_warning "Manual SSH key deployment required for Mac Studio $studio"
            print_info "Run: ssh-copy-id -i $SSH_KEY_PATH ${SSH_USER}@${studio}"
        fi
    done
    
    # Deploy to Mac Minis
    for mini in "${MAC_MINIS[@]}"; do
        print_info "Deploying SSH key to Mac Mini $mini..."
        
        if ssh-copy-id -i "$SSH_KEY_PATH" "${SSH_USER}@${mini}" 2>/dev/null; then
            print_status "SSH key deployed to Mac Mini $mini"
        else
            print_warning "Manual SSH key deployment required for Mac Mini $mini"
            print_info "Run: ssh-copy-id -i $SSH_KEY_PATH ${SSH_USER}@${mini}"
        fi
    done
}

# Configure remote nodes
configure_remote_nodes() {
    print_info "Configuring remote nodes..."
    
    # Create Ansible inventory
    cat > ~/.config/master-orchestrator/inventory.yml << EOF
all:
  children:
    mac_studios:
      hosts:
$(for studio in "${MAC_STUDIOS[@]}"; do echo "        mac-studio-${studio##*.}:"; echo "          ansible_host: $studio"; done)
    mac_minis:
      hosts:
$(for mini in "${MAC_MINIS[@]}"; do echo "        mac-mini-${mini##*.}:"; echo "          ansible_host: $mini"; done)
    nas_systems:
      hosts:
$(for nas in "${NAS_SYSTEMS[@]}"; do echo "        nas-${nas##*.}:"; echo "          ansible_host: $nas"; done)
  vars:
    ansible_user: $SSH_USER
    ansible_ssh_private_key_file: $SSH_KEY_PATH
    ansible_ssh_common_args: '-o StrictHostKeyChecking=no'
EOF

    # Create Ansible playbook for node configuration
    cat > ~/.config/master-orchestrator/configure_nodes.yml << 'EOF'
---
- name: Configure Master Orchestrator Nodes
  hosts: all
  become: yes
  
  tasks:
    - name: Update system packages (macOS)
      shell: softwareupdate -i -a
      when: ansible_os_family == "Darwin"
      ignore_errors: yes
    
    - name: Install Homebrew (macOS)
      shell: /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
      when: ansible_os_family == "Darwin"
      become_user: "{{ ansible_user }}"
      ignore_errors: yes
    
    - name: Install required packages (macOS)
      homebrew:
        name:
          - python3
          - docker
          - htop
          - wget
          - curl
        state: present
      when: ansible_os_family == "Darwin"
      become_user: "{{ ansible_user }}"
      ignore_errors: yes
    
    - name: Create orchestrator directories
      file:
        path: "{{ item }}"
        state: directory
        mode: '0755'
        owner: "{{ ansible_user }}"
      loop:
        - /opt/master-orchestrator
        - /opt/master-orchestrator/data
        - /opt/master-orchestrator/logs
        - /opt/master-orchestrator/cache
    
    - name: Install Python dependencies
      pip:
        name:
          - psutil
          - docker
          - requests
        state: present
      become_user: "{{ ansible_user }}"
    
    - name: Create monitoring script
      copy:
        content: |
          #!/usr/bin/env python3
          import psutil
          import json
          import time
          from datetime import datetime
          
          def get_system_metrics():
              return {
                  'timestamp': datetime.utcnow().isoformat(),
                  'hostname': psutil.hostname(),
                  'cpu_percent': psutil.cpu_percent(interval=1),
                  'memory_percent': psutil.virtual_memory().percent,
                  'disk_percent': psutil.disk_usage('/').percent,
                  'load_average': psutil.getloadavg(),
                  'uptime': time.time() - psutil.boot_time()
              }
          
          if __name__ == '__main__':
              print(json.dumps(get_system_metrics()))
        dest: /opt/master-orchestrator/system_metrics.py
        mode: '0755'
        owner: "{{ ansible_user }}"
    
    - name: Create log rotation config
      copy:
        content: |
          /opt/master-orchestrator/logs/*.log {
              daily
              missingok
              rotate 7
              compress
              delaycompress
              notifempty
              copytruncate
          }
        dest: /etc/logrotate.d/master-orchestrator
        mode: '0644'
      ignore_errors: yes

- name: Configure Mac Studios for compute workloads
  hosts: mac_studios
  become: yes
  
  tasks:
    - name: Configure high performance mode
      shell: |
        # Disable sleep
        sudo pmset -a sleep 0
        sudo pmset -a displaysleep 10
        sudo pmset -a disksleep 0
        
        # Optimize for performance
        sudo sysctl -w kern.maxproc=4096
        sudo sysctl -w kern.maxprocperuid=2048
      ignore_errors: yes
    
    - name: Start Docker service
      shell: open -a Docker
      become_user: "{{ ansible_user }}"
      ignore_errors: yes

- name: Configure Mac Minis for edge computing
  hosts: mac_minis
  become: yes
  
  tasks:
    - name: Configure power management
      shell: |
        # Optimized power settings for 24/7 operation
        sudo pmset -a sleep 0
        sudo pmset -a displaysleep 5
        sudo pmset -a disksleep 10
      ignore_errors: yes
    
    - name: Setup monitoring cron job
      cron:
        name: "System metrics collection"
        minute: "*/5"
        job: "/opt/master-orchestrator/system_metrics.py >> /opt/master-orchestrator/logs/metrics.log"
        user: "{{ ansible_user }}"
EOF

    # Run Ansible playbook
    print_info "Running Ansible configuration playbook..."
    if ansible-playbook -i ~/.config/master-orchestrator/inventory.yml ~/.config/master-orchestrator/configure_nodes.yml; then
        print_status "Remote node configuration completed"
    else
        print_warning "Some configuration tasks may have failed - check manually"
    fi
}

# Setup network monitoring
setup_monitoring() {
    print_info "Setting up network and hardware monitoring..."
    
    # Create monitoring script
    cat > ~/.config/master-orchestrator/monitor_network.sh << EOF
#!/bin/bash

# Master Orchestrator Network Monitor
# Monitors all nodes and reports status

NODES=(${MAC_STUDIOS[@]} ${MAC_MINIS[@]} ${NAS_SYSTEMS[@]})
LOG_FILE=~/.config/master-orchestrator/monitoring/network_status.log

echo "\$(date): Starting network monitoring sweep" >> \$LOG_FILE

for node in "\${NODES[@]}"; do
    if ping -c 1 -W 2000 "\$node" &> /dev/null; then
        echo "\$(date): \$node - UP" >> \$LOG_FILE
    else
        echo "\$(date): \$node - DOWN" >> \$LOG_FILE
    fi
done

echo "\$(date): Network monitoring sweep completed" >> \$LOG_FILE
EOF

    chmod +x ~/.config/master-orchestrator/monitor_network.sh
    
    # Setup cron job for network monitoring
    (crontab -l 2>/dev/null; echo "*/5 * * * * ~/.config/master-orchestrator/monitor_network.sh") | crontab -
    
    print_status "Network monitoring configured"
}

# Create hardware inventory
create_hardware_inventory() {
    print_info "Creating hardware inventory..."
    
    cat > ~/.config/master-orchestrator/hardware_inventory.json << EOF
{
  "control_center": {
    "hostname": "$(hostname)",
    "type": "macbook_pro_m4_max",
    "ram_gb": 128,
    "role": "orchestrator",
    "ip_address": "$(ipconfig getifaddr en0 2>/dev/null || echo 'unknown')"
  },
  "mac_studios": [
$(for i in "${!MAC_STUDIOS[@]}"; do
    echo "    {"
    echo "      \"name\": \"mac-studio-$((i+1))\","
    echo "      \"ip_address\": \"${MAC_STUDIOS[i]}\","
    echo "      \"ram_gb\": 512,"
    echo "      \"cpu\": \"Apple Silicon\","
    echo "      \"role\": \"compute\""
    if [ $i -lt $((${#MAC_STUDIOS[@]}-1)) ]; then echo "    },"; else echo "    }"; fi
done)
  ],
  "mac_minis": [
$(for i in "${!MAC_MINIS[@]}"; do
    echo "    {"
    echo "      \"name\": \"mac-mini-$((i+1))\","
    echo "      \"ip_address\": \"${MAC_MINIS[i]}\","
    echo "      \"ram_gb\": 64,"
    echo "      \"cpu\": \"Apple M4 Max\","
    echo "      \"role\": \"edge_compute\""
    if [ $i -lt $((${#MAC_MINIS[@]}-1)) ]; then echo "    },"; else echo "    }"; fi
done)
  ],
  "nas_systems": [
$(for i in "${!NAS_SYSTEMS[@]}"; do
    echo "    {"
    echo "      \"name\": \"nas-$((i+1))\","
    echo "      \"ip_address\": \"${NAS_SYSTEMS[i]}\","
    echo "      \"storage_tb\": \"$([[ $i -eq 0 ]] && echo 1000 || echo 100)\","
    echo "      \"type\": \"$([[ $i -eq 0 ]] && echo 'synology' || echo 'asustor')\","
    echo "      \"role\": \"storage\""
    if [ $i -lt $((${#NAS_SYSTEMS[@]}-1)) ]; then echo "    },"; else echo "    }"; fi
done)
  ],
  "network": {
    "type": "10gb_l3_mesh",
    "backbone": "thunderbolt",
    "total_nodes": $((${#MAC_STUDIOS[@]} + ${#MAC_MINIS[@]} + ${#NAS_SYSTEMS[@]} + 1))
  }
}
EOF
    
    print_status "Hardware inventory created"
}

# Setup Thunderbolt network optimization
setup_thunderbolt_network() {
    print_info "Configuring Thunderbolt network optimization..."
    
    # Check for Thunderbolt connections
    if system_profiler SPThunderboltDataType | grep -q "Thunderbolt"; then
        print_status "Thunderbolt connections detected"
        
        # Create network optimization script
        cat > ~/.config/master-orchestrator/optimize_network.sh << 'EOF'
#!/bin/bash

# Thunderbolt Network Optimization
echo "Optimizing Thunderbolt network settings..."

# Increase network buffer sizes
sudo sysctl -w net.inet.tcp.sendspace=1048576
sudo sysctl -w net.inet.tcp.recvspace=1048576
sudo sysctl -w net.inet.udp.maxdgram=65536

# Optimize for high-bandwidth, low-latency
sudo sysctl -w net.inet.tcp.delayed_ack=0
sudo sysctl -w net.inet.tcp.nagle_limit=1

echo "Network optimization complete"
EOF
        
        chmod +x ~/.config/master-orchestrator/optimize_network.sh
        ~/.config/master-orchestrator/optimize_network.sh
        
    else
        print_warning "No Thunderbolt connections detected - using standard Ethernet"
    fi
}

# Create deployment verification script
create_verification_script() {
    print_info "Creating deployment verification script..."
    
    cat > ~/.config/master-orchestrator/verify_deployment.sh << 'EOF'
#!/bin/bash

# Master Orchestrator Deployment Verification

echo "🔍 Master Orchestrator Hardware Deployment Verification"
echo "======================================================="

# Test SSH connectivity to all nodes
echo "Testing SSH connectivity..."
ansible all -i ~/.config/master-orchestrator/inventory.yml -m ping

# Check system resources on all nodes
echo "Checking system resources..."
ansible all -i ~/.config/master-orchestrator/inventory.yml -m shell -a "python3 /opt/master-orchestrator/system_metrics.py"

# Verify Docker installation on compute nodes
echo "Verifying Docker installation..."
ansible mac_studios:mac_minis -i ~/.config/master-orchestrator/inventory.yml -m shell -a "docker --version"

# Test network performance between nodes
echo "Testing network performance..."
ansible mac_studios -i ~/.config/master-orchestrator/inventory.yml -m shell -a "iperf3 -c $(ipconfig getifaddr en0) -t 10 -P 4" --limit 1

echo "✅ Verification complete"
EOF
    
    chmod +x ~/.config/master-orchestrator/verify_deployment.sh
}

# Main setup function
main() {
    echo "🚀 Starting Master Orchestrator Hardware Setup"
    echo "==============================================="
    echo
    
    # Check if running on macOS
    if [[ "$OSTYPE" != "darwin"* ]]; then
        print_warning "This script is optimized for macOS environments"
        print_info "Some features may not work on other systems"
    fi
    
    setup_ssh_keys
    setup_control_center
    test_connectivity
    deploy_ssh_keys
    configure_remote_nodes
    setup_monitoring
    create_hardware_inventory
    setup_thunderbolt_network
    create_verification_script
    
    echo
    print_status "Hardware setup completed successfully!"
    echo
    print_info "Next steps:"
    echo "  1. Verify deployment: ~/.config/master-orchestrator/verify_deployment.sh"
    echo "  2. Update Master Orchestrator config.yaml with node IP addresses"
    echo "  3. Start Master Orchestrator: ./quick_start.sh"
    echo "  4. Monitor hardware: ~/.config/master-orchestrator/monitor_network.sh"
    echo
    print_info "Hardware inventory: ~/.config/master-orchestrator/hardware_inventory.json"
    print_info "Monitoring logs: ~/.config/master-orchestrator/monitoring/"
    echo
}

# Run main setup
main "$@"