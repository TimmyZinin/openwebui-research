#!/bin/bash
# Upload finished video to RUVDS for social media publishing
# Usage: ./upload_to_ruvds.sh /path/to/video.mp4 /path/to/metadata.json

set -e

VIDEO="$1"
METADATA="$2"

if [ -z "$VIDEO" ] || [ -z "$METADATA" ]; then
    echo "Usage: $0 <video.mp4> <metadata.json>"
    exit 1
fi

REMOTE_DIR="/opt/zinin-corp/data/lisa/videos/pending"
RUVDS="root@88.218.248.114"
SSH_OPTS="-o PubkeyAuthentication=no -o PreferredAuthentications=password"

echo "[Upload] Uploading video to RUVDS..."

# Create remote dir + upload video + metadata in one session
sshpass -p 'Ruvds2026' ssh $SSH_OPTS $RUVDS "mkdir -p $REMOTE_DIR"

sshpass -p 'Ruvds2026' scp $SSH_OPTS "$VIDEO" "$RUVDS:$REMOTE_DIR/"
sshpass -p 'Ruvds2026' scp $SSH_OPTS "$METADATA" "$RUVDS:$REMOTE_DIR/"

echo "[Upload] Done. Files uploaded to $REMOTE_DIR"
echo "[Upload] Auto-publisher on RUVDS will pick them up."
