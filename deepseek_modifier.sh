#!/bin/bash

# Secondary script to run DeepSeek (qwen2.5-7b-instruct-1m) for file modifications based on lm_env.log
echo "Starting DeepSeek Modifier (Secondary Process)..."

# Ensure we're in the project directory
cd ~/aios_project || { echo "Failed to enter ~/aios_project"; exit 1; }

# Named pipe for escape commands
FIFO="$HOME/aios_project/escape_fifo"
# Log files
ENV_LOG="$HOME/aios_project/lm_env.log"
TOOL_LOG="$HOME/aios_project/tool_changes.log"

# Cleanup function for the secondary process
cleanup() {
    echo "Cleaning up DeepSeek Modifier (Secondary Process)..."
    rm -f "$DEEPSEEK_PID_FILE"
    exit 0
}

# Trap Ctrl+C (SIGINT) and other signals to ensure cleanup
trap cleanup SIGINT SIGTERM EXIT

# Step 1: Create named pipe for escape commands if not already created by primary process
if [ ! -p "$FIFO" ]; then
    mkfifo "$FIFO" || echo "Named pipe already exists, proceeding..."
fi

# Step 2: Persistent loop to monitor lm_env.log and apply modifications
echo "DeepSeek Modifier: Monitoring lm_env.log for environmental feedback..."
last_line_processed=""
while true; do
    # Read the latest line from lm_env.log
    latest_line=$(tail -n 1 "$ENV_LOG")
    if [ "$latest_line" != "$last_line_processed" ] && [ -n "$latest_line" ]; then
        # Process the latest feedback
        echo "DeepSeek Modifier: Processing new feedback: $latest_line"
        # Send the feedback to DeepSeek for modification
        python3 - <<END
import requests
import json
import os
import ast
import time

def modify_file(file_path, content, mode, tool_log):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            before_content = f.read()
    else:
        before_content = "File does not exist."
    # Validate the content as valid Python if modifying ai_model.py
    if file_path == "ai_model.py":
        try:
            ast.parse(content)
        except SyntaxError as e:
            print(f"Invalid Python code in modification for {file_path}: {e}")
            return
    if os.path.exists(file_path):
        if mode == "overwrite":
            with open(file_path, "w") as f:
                f.write(content)
        elif mode == "append":
            with open(file_path, "a") as f:
                f.write(content + "\n")
        # Log the change details with timestamp
        with open(tool_log, "a") as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n=== Tool Change at {timestamp} ===\n")
            f.write(f"File: {file_path}\n")
            f.write(f"Mode: {mode}\n")
            f.write(f"Before Modification:\n{before_content}\n")
            f.write(f"Content Applied:\n{content}\n")
            f.write("================\n")
        print(f"DeepSeek Modifier: Applied modification to {file_path} in mode {mode} at {timestamp}. Check tool_changes.log for details.")
    else:
        print(f"DeepSeek Modifier: Warning: File {file_path} does not exist, skipping modification.")

# Fetch the latest feedback
with open("$ENV_LOG", "r") as f:
    lines = f.readlines()
    latest_feedback = lines[-1].strip() if lines else ""

# Send feedback to DeepSeek for processing
prompt = f"Latest environmental feedback: {latest_feedback}\nBased on this feedback, suggest improvements to the QEMU environment or AIOS core code in boot.asm to handle the command. Use the 'modify_file' tool to suggest changes."

MODIFY_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "modify_file",
        "description": "Modify a file with new content or append code modifications.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file (e.g., ai_model.py, boot.asm)"},
                "content": {"type": "string", "description": "New content or code to append"},
                "mode": {"type": "string", "enum": ["overwrite", "append"], "description": "Mode to modify the file"}
            },
            "required": ["file_path", "content", "mode"]
        }
    }
}

payload = {
    "messages": [
        {"role": "system", "content": "You are the System Commander (qwen2.5-7b-instruct-1m), an expert system designer improving the QEMU simulation environment for a custom AI-driven Operating System (AIOS). Use the 'modify_file' tool to suggest file changes."},
        {"role": "user", "content": prompt}
    ],
    "max_tokens": 200,
    "temperature": 0.7,
    "model": "qwen2.5-7b-instruct-1m",
    "tools": [MODIFY_FILE_TOOL]
}

headers = {"Content-Type": "application/json", "Authorization": "Bearer lm-studio"}
try:
    resp = requests.post("http://10.150.1.8:4891/v1/chat/completions", json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    response = resp.json()["choices"][0]["message"]
    if "tool_calls" in response:
        for tool_call in response["tool_calls"]:
            if tool_call["function"]["name"] == "modify_file":
                args = json.loads(tool_call["function"]["arguments"])
                file_path = args["file_path"]
                content = args["content"]
                mode = args["mode"]
                modify_file(file_path, content, mode, "$TOOL_LOG")
except requests.exceptions.RequestException as e:
    print(f"DeepSeek Modifier: LM-Studio connection failed: {str(e)}")
END
        last_line_processed="$latest_line"
    fi
    # Check for escape commands via named pipe
    if [ -p "$FIFO" ]; then
        if read -t 2 command < "$FIFO"; then
            case $command in
                "quit")
                    echo "DeepSeek Modifier: Received 'quit' command, stopping..."
                    cleanup
                    ;;
                "stop_deepseek")
                    echo "DeepSeek Modifier: Received 'stop_deepseek' command, stopping..."
                    cleanup
                    ;;
            esac
        fi
    fi
    sleep 2  # Short delay to prevent overwhelming the system
done
