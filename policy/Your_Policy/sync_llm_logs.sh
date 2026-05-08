#!/bin/bash
# Sync LLM logs from remote server to local machine
REMOTE_HOST="ubuntu"
REMOTE_DIR="~/RoboTwin-release/policy/Your_Policy/logs"
LOCAL_DIR="./policy/Your_Policy/logs"

mkdir -p "$LOCAL_DIR"
echo "Syncing logs from $REMOTE_HOST..."
rsync -avP "$REMOTE_HOST:$REMOTE_DIR/" "$LOCAL_DIR/"
echo "Done."
