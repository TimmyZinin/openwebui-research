#!/bin/bash
# AI Video Factory — VPS Setup Script
# Contabo VPS 10: 4 vCPU / 8 GB RAM / 75 GB NVMe / Ubuntu 24.04
# Run as root: bash setup_vps.sh

set -e

echo "=== AI Video Factory — VPS Setup ==="
echo "Date: $(date)"
echo ""

# 1. System update
echo "[1/8] Updating system..."
apt update && apt upgrade -y

# 2. Install essentials
echo "[2/8] Installing essentials..."
apt install -y \
    curl wget git htop tmux unzip \
    python3 python3-pip python3-venv \
    ffmpeg \
    nodejs npm \
    build-essential

# 3. Install Node.js 20 LTS (for OpenClaw)
echo "[3/8] Installing Node.js 20 LTS..."
curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt install -y nodejs
npm install -g npm@latest

# 4. Install OpenClaw
echo "[4/8] Installing OpenClaw..."
npm install -g openclaw@latest

# 5. Setup Python environment for video processing
echo "[5/8] Setting up Python video environment..."
python3 -m venv /opt/ai-video/venv
source /opt/ai-video/venv/bin/activate
pip install --upgrade pip
pip install \
    moviepy \
    pillow \
    requests \
    python-dotenv \
    pexels-api \
    yt-dlp

# 6. Create project structure
echo "[6/8] Creating project structure..."
mkdir -p /opt/ai-video/{config,scripts,output,assets,logs}
mkdir -p /opt/ai-video/assets/{music,broll,avatars,fonts}

# 7. Create .env template
echo "[7/8] Creating config..."
cat > /opt/ai-video/config/.env << 'ENVEOF'
# AI Video Factory Configuration
# LLM Providers
DEEPSEEK_API_KEY=
GLM_API_KEY=
GROQ_API_KEY=

# Image Generation
GEMINI_API_KEY=

# Video Sources
PEXELS_API_KEY=

# Telegram
TELEGRAM_BOT_TOKEN=

# HeyGen (session cookies, added later)
HEYGEN_SESSION=
ENVEOF

# 8. System tuning
echo "[8/8] System tuning..."
# Increase file limits for video processing
echo "* soft nofile 65535" >> /etc/security/limits.conf
echo "* hard nofile 65535" >> /etc/security/limits.conf

# Enable swap (2GB) for safety with 8GB RAM
fallocate -l 2G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab

# Setup firewall
ufw allow ssh
ufw allow 80
ufw allow 443
ufw --force enable

echo ""
echo "=== Setup Complete ==="
echo "OpenClaw: $(openclaw --version 2>/dev/null || echo 'installed')"
echo "Node.js: $(node --version)"
echo "Python: $(python3 --version)"
echo "FFmpeg: $(ffmpeg -version 2>&1 | head -1)"
echo ""
echo "Next steps:"
echo "1. Fill /opt/ai-video/config/.env with API keys"
echo "2. Configure OpenClaw with LLM provider"
echo "3. Upload Lisa photos to /opt/ai-video/assets/avatars/"
echo "4. Start video pipeline"
