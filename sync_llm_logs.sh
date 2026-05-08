#!/bin/bash
# Sync LLM Agent logs from remote server to local
# Run from RoboTwin repo root on your local machine

REMOTE_LOG_DIR="ubuntu:~/RoboTwin-release/policy/Your_Policy/logs/"
LOCAL_LOG_DIR="./policy/Your_Policy/logs/"

mkdir -p "$LOCAL_LOG_DIR"

echo "Syncing logs from server..."
rsync -avP "$REMOTE_LOG_DIR" "$LOCAL_LOG_DIR"

echo ""
echo "Latest log files:"
ls -lt "$LOCAL_LOG_DIR" | head -n 10
