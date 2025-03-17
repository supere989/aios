#!/bin/bash

# Primary script to manage the AIOS workflow with persistent loop, dual LM Studio models, and integrated DeepSeek control
echo "Starting AIOS workflow (Primary Process) with persistent loop and dual LM Studio models..."

# Configuration for LM-Studio URLs (can be overridden with environment variables)
export LM_STUDIO_URL=${LM_STUDIO_URL:-"http://10.150.1.8:4891/v1/chat/completions"}
export LM_STUDIO_BACKUP_URL=${LM_STUDIO_BACKUP_URL:-"http://localhost:4891/v1/chat/completions"}

# Configuration for retry mechanism
export LM_RETRY_COUNT=${LM_RETRY_COUNT:-3}
export LM_RETRY_BACKOFF=${LM_RETRY_BACKOFF:-2}
export LM_TIMEOUT=${LM_TIMEOUT:-30}

# Logging configuration
export LOG_LEVEL=${LOG_LEVEL:-"INFO"} # ERROR, WARN, INFO, DEBUG
export LOG_TIMESTAMP=${LOG_TIMESTAMP:-true}

# Print configuration
echo "Configuration:"
echo "  LM_STUDIO_URL: $LM_STUDIO_URL"
echo "  LM_STUDIO_BACKUP_URL: $LM_STUDIO_BACKUP_URL"
echo "  LM_RETRY_COUNT: $LM_RETRY_COUNT"
echo "  LM_RETRY_BACKOFF: $LM_RETRY_BACKOFF"
echo "  LM_TIMEOUT: $LM_TIMEOUT"
echo "  LOG_LEVEL: $LOG_LEVEL"
echo "  LOG_TIMESTAMP: $LOG_TIMESTAMP"

# Ensure we're in the project directory
cd ~/aios_project || { echo "Failed to enter ~/aios_project"; exit 1; }

# File to store QEMU PID
QEMU_PID_FILE="$HOME/aios_project/qemu.pid"
# Named pipe for escape commands
FIFO="$HOME/aios_project/escape_fifo"
# Log files for QEMU, LM Studio training guidance, and environmental updates
QEMU_LOG="$HOME/aios_project/qemu_output.log"
LM_GUIDANCE_LOG="$HOME/aios_project/lm_guidance.log"
LM_ENV_LOG="$HOME/aios_project/lm_env.log"
# Log file for tool changes
TOOL_LOG="$HOME/aios_project/tool_changes.log"

# Flag to control DeepSeek modification task
DEEPSEEK_RUNNING=1
DEEPSEEK_PID=""

# Cleanup function to stop QEMU and DeepSeek task
cleanup() {
    echo "Cleaning up QEMU environment (Primary Process)..."
    if [ -f "$QEMU_PID_FILE" ]; then
        QEMU_PID=$(cat "$QEMU_PID_FILE")
        if ps -p $QEMU_PID > /dev/null 2>&1; then
            echo "Stopping QEMU (PID: $QEMU_PID)..."
            kill $QEMU_PID
            sleep 1
            if ps -p $QEMU_PID > /dev/null 2>&1; then
                echo "QEMU did not stop gracefully, forcing termination..."
                kill -9 $QEMU_PID
            fi
        fi
        rm -f "$QEMU_PID_FILE"
    fi
    # Stop the DeepSeek task if running
    if [ "$DEEPSEEK_RUNNING" -eq 1 ] && [ -n "$DEEPSEEK_PID" ]; then
        if ps -p $DEEPSEEK_PID > /dev/null 2>&1; then
            echo "Stopping DeepSeek task (PID: $DEEPSEEK_PID)..."
            kill $DEEPSEEK_PID
            sleep 1
            if ps -p $DEEPSEEK_PID > /dev/null 2>&1; then
                echo "DeepSeek task did not stop gracefully, forcing termination..."
                kill -9 $DEEPSEEK_PID
            fi
        fi
        DEEPSEEK_PID=""
    fi
    # Remove the named pipe
    rm -f "$FIFO"
    exit 0
}

# Trap Ctrl+C (SIGINT) and other signals to ensure cleanup
trap cleanup SIGINT SIGTERM EXIT

# Step 1: Clean up any existing QEMU instances
if [ -f "$QEMU_PID_FILE" ]; then
    QEMU_PID=$(cat "$QEMU_PID_FILE")
    if ps -p $QEMU_PID > /dev/null 2>&1; then
        echo "Found existing QEMU instance (PID: $QEMU_PID), stopping it..."
        kill $QEMU_PID
        sleep 1
        if ps -p $QEMU_PID > /dev/null 2>&1; then
            echo "QEMU did not stop gracefully, forcing termination..."
            kill -9 $QEMU_PID
        fi
    fi
    rm -f "$QEMU_PID_FILE"
fi

# Step 2: Start QEMU with the latest aios.bin
echo "Starting QEMU with updated AIOS core..."
qemu-system-x86_64 -drive file=aios.bin,format=raw -monitor tcp:localhost:4444,server,nowait -m 512M &
QEMU_PID=$!
sleep 2
if ! ps -p $QEMU_PID > /dev/null; then
    echo "Failed to start QEMU"
    exit 1
fi
echo $QEMU_PID > "$QEMU_PID_FILE"
echo "QEMU started with PID: $QEMU_PID"

# Step 3: Initialize log files
echo "Initializing log files..."
: > "$QEMU_LOG"
: > "$LM_GUIDANCE_LOG"
: > "$LM_ENV_LOG"
: > "$TOOL_LOG"

# Step 4: Start the DeepSeek modification task in the background
deepseek_task() {
    echo "DeepSeek Modifier: Monitoring lm_env.log for environmental feedback..."
    last_line_processed=""
    while [ "$DEEPSEEK_RUNNING" -eq 1 ]; do
        # Read the latest line from lm_env.log
        latest_line=$(tail -n 1 "$LM_ENV_LOG" 2>/dev/null || echo "")
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
import re

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
with open("$LM_ENV_LOG", "r") as f:
    lines = f.readlines()
    latest_feedback = lines[-1].strip() if lines else ""

# Parse the feedback for file modification instructions
# Look for patterns like: "modify file [file_path] with content [content] in mode [mode]"
file_path_pattern = r"modify file (\S+)"
content_pattern = r"with content (.*?)(?=( in mode |$))"
mode_pattern = r"in mode (overwrite|append)"

file_path_match = re.search(file_path_pattern, latest_feedback)
content_match = re.search(content_pattern, latest_feedback)
mode_match = re.search(mode_pattern, latest_feedback)

if file_path_match and content_match and mode_match:
    file_path = file_path_match.group(1)
    content = content_match.group(1).strip()
    mode = mode_match.group(1)
    modify_file(file_path, content, mode, "$TOOL_LOG")
else:
    print(f"DeepSeek Modifier: No valid modification instructions found in feedback: {latest_feedback}")

END
            last_line_processed="$latest_line"
        fi
        sleep 2  # Short delay to prevent overwhelming the system
    done
} &
DEEPSEEK_PID=$!
echo "DeepSeek task started with PID: $DEEPSEEK_PID"

# Step 5: Update ai_model.py with real QEMU responses, dual model chat, and detailed tool logging
echo "Updating ai_model.py with real QEMU responses, retry mechanism, and detailed tool logging..."
# cat << 'EOF' > ai_model.py  # Commented out to preserve the fixed version of ai_model.py
# The following heredoc is skipped to avoid overwriting the fixed ai_model.py file that resolves the syntax error
import socket
import time
import torch
import torch.nn as nn
import requests
import json
import os
import ast
import logging
import sys
from requests.exceptions import RequestException, Timeout, ConnectionError

# Class to communicate with QEMU's monitor interface
class QEMUComm:
    def __init__(self, host="localhost", port=4444, log_file="qemu_output.log"):
        self.host = host
        self.port = port
        self.log_file = log_file

    def send_command(self, cmd):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.host, self.port))
            sock.send((cmd + "\n").encode())
            time.sleep(0.1)
            response = sock.recv(4096).decode().strip()
            sock.close()
            # Log to QEMU output file
            with open(self.log_file, "a") as f:
                f.write(f"Command: {cmd}\nResponse: {response}\n---\n")
            return response
        except Exception as e:
            response = f"Communication error: {str(e)}"
            with open(self.log_file, "a") as f:
                f.write(f"Command: {cmd}\nResponse: {response}\n---\n")
            return response

# Simple neural network for generating commands
class AIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 2)  # Input: [intent, size_mb, free_mb, 0, 0], Output: [op, size]

    def forward(self, x):
        return torch.relu(self.fc(x))  # Ensure non-negative outputs

# Main controller for AIOS AI with dual model chat and tool use
class AIOSController:
    def __init__(self):
        # Configure logging
        log_level = os.environ.get("LOG_LEVEL", "INFO")
        log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARN": logging.WARNING,
            "ERROR": logging.ERROR
        }
        logging.basicConfig(
            level=log_levels.get(log_level, logging.INFO),
            format='%(asctime)s [%(levelname)s] %(message)s' if os.environ.get("LOG_TIMESTAMP", "true").lower() == "true" else '[%(levelname)s] %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("aios.log")
            ]
        )
        self.logger = logging.getLogger("AIOSController")
        self.logger.info("Initializing AIOSController with configured logging")
        
        self.model = AIModel()
        self.comm = QEMUComm()
        self.intent_map = {"allocate": 1}  # Focus on allocate only
        
        # LM Studio configuration from environment variables
        self.lm_url = os.environ.get("LM_STUDIO_URL", "http://10.150.1.8:4891/v1/chat/completions")
        self.lm_backup_url = os.environ.get("LM_STUDIO_BACKUP_URL", "http://localhost:4891/v1/chat/completions")
        self.lm_retry_count = int(os.environ.get("LM_RETRY_COUNT", 3))
        self.lm_retry_backoff = int(os.environ.get("LM_RETRY_BACKOFF", 2))
        self.lm_timeout = int(os.environ.get("LM_TIMEOUT", 30))
        
        self.lm_guidance_log = "lm_guidance.log"
        self.lm_env_log = "lm_env.log"
        self.tool_log = "tool_changes.log"
        self.logger.info(f"Using LM Studio URL: {self.lm_url} (backup: {self.lm_backup_url})")
        # Initialize chat history for dual model interaction
        self.chat_history = [
            {"role": "system", "content": "<start_of_turn>system\nThis is a collaborative chat between the Trainer (gemma-3-4b-it) and System Commander (qwen2.5-7b-instruct-1m) to improve the AIOS system.<end_of_turn>\n"}
        ]

    def process_intent(self, intent, size_mb, free_mb):
        intent_code = self.intent_map.get(intent, 0)
        if intent_code == 0:
            return "Invalid intent", "Error"
        input_tensor = torch.tensor([intent_code, size_mb, free_mb, 0, 0], dtype=torch.float32)
        output = self.model(input_tensor)
        op, size = [max(0, int(x)) for x in output.tolist()]
        size = max(4, min(4096, size))  # Ensure size is between 4 and 4096 bytes
        cmd = f"alloc {size}B"  # Add "B" for bytes
        response = self.comm.send_command(cmd)  # Use real QEMU response
        return cmd, response

    def query_lm_studio(self, prompt, role="training", model=None, use_tools=False):
        # Update chat history with appropriate template
        if role == "training":
            formatted_prompt = "<start_of_turn>user\n" + prompt + "<end_of_turn>\n"
            self.chat_history.append({"role": "user", "content": formatted_prompt})
        else:
            formatted_prompt = prompt
            self.chat_history.append({"role": "user", "content": formatted_prompt})

        # Dual roles: Training Guidance (gemma-3-4b-it) and Environmental Updates (qwen2.5-7b-instruct-1m)
        system_content = ""
        if role == "training":
            system_content = "<start_of_turn>system\nYou are the Trainer (gemma-3-4b-it), an expert AI trainer improving the training logic for a custom AI-driven Operating System (AIOS) running in a QEMU simulation. Your task is to evaluate AI-generated commands and their responses, suggest improvements to the training process (e.g., reward system, model architecture), and provide Python code modifications to enhance the AI model in ai_model.py. Be concise and technically accurate.<end_of_turn>\n"
            model = "gemma-3-4b-it"
            log_file = self.lm_guidance_log
            stop = ["<end_of_turn>"]
        elif role == "environment":
            model = "qwen2.5-7b-instruct-1m"
            log_file = self.lm_env_log
            stop = ["<end_of_turn>"]

        payload = {
            "messages": [{"role": "system", "content": system_content}] + self.chat_history if role == "training" else self.chat_history,
            "max_tokens": 200,
            "temperature": 0.7,
            "model": model,
            "stop": stop
        }
        # Temporarily disable tools to resolve 400 error
        # if use_tools:
        #     payload["tools"] = [MODIFY_FILE_TOOL]

        headers = {"Content-Type": "application/json", "Authorization": "Bearer lm-studio"}
        
        # Initialize retry mechanism
        retry_count = self.lm_retry_count
        backoff_time = 1  # Start with 1 second
        urls_to_try = [self.lm_url]
        if self.lm_url != self.lm_backup_url:
            urls_to_try.append(self.lm_backup_url)
        
        for current_url in urls_to_try:
            current_retry = 0
            while current_retry <= retry_count:
                try:
                    self.logger.info(f"Attempting connection to {current_url}, attempt {current_retry+1}/{retry_count+1}")
                    resp = requests.post(current_url, json=payload, headers=headers, timeout=self.lm_timeout)
                    resp.raise_for_status()
                    response = resp.json()["choices"][0]["message"]
                    content_to_log = response["content"].strip() if "content" in response else "No content in response."
                    # Append response to chat history with proper formatting
                    if role == "training":
                        formatted_response = "<start_of_turn>model\n" + content_to_log + "<end_of_turn>\n"
                    else:
                        formatted_response = "<start_of_turn>assistant\n" + content_to_log + "<end_of_turn>\n"
                    self.chat_history.append({"role": "assistant", "content": formatted_response})
                    # Log to the appropriate file
                    with open(log_file, "a") as f:
                        f.write(f"Prompt: {prompt}\nFeedback: {content_to_log}\n---\n")
                    self.logger.info(f"Successfully received response from {current_url}")
                    return content_to_log
                except (ConnectionError, Timeout) as e:
                    self.logger.warning(f"Connection error to {current_url}: {str(e)}. Retry {current_retry+1}/{retry_count+1}")
                    if current_retry < retry_count:
                        sleep_time = backoff_time * (self.lm_retry_backoff ** current

    def train_step(self, intent, size_mb, free_mb):
        cmd, response = self.process_intent(intent, size_mb, free_mb)
        # Step 1: Trainer evaluates the command and response
        train_prompt = f"Evaluate command '{cmd}' with response '{response}'. Is it correct? Suggest improvements to the training process or AI model code in ai_model.py."
        train_response = self.query_lm_studio(train_prompt, "training", model="gemma-3-4b-it", use_tools=False)
        # Step 2: System Commander evaluates based on Trainer's feedback
        env_prompt = f"Trainer Feedback: {train_response}\nEvaluate command '{cmd}' with response '{response}'. Suggest improvements to the QEMU environment or AIOS core code in boot.asm to better handle this command."
        env_resp = self.query_lm_studio(env_prompt, "environment", model="qwen2.5-7b-instruct-1m", use_tools=False)
        # Step 3: Trainer responds to System Commander's updates
        train_prompt_2 = f"System Commander Feedback: {env_resp}\nEvaluate command '{cmd}' with response '{response}'. Suggest further improvements to the training process or AI model code in ai_model.py."
        train_response_2 = self.query_lm_studio(train_prompt_2, "training", model="gemma-3-4b-it", use_tools=False)
        if "incorrect" in train_response.lower() or "incorrect" in train_response_2.lower():
            reward = -1
        elif "correct" in train_response.lower() or "correct" in train_response_2.lower():
            reward = 1
        else:
            reward = 0
        print(f"Reward: {reward} (Initial Training Feedback: {train_response})")
        print(f"Environmental Feedback: {env_resp}")
        print(f"Follow-up Training Feedback: {train_response_2}")
        return cmd, response, train_response, env_resp, train_response_2

# Persistent loop to run the training and allow LM Studio models to collaborate
if __name__ == "__main__":
    ai = AIOSController()
    while True:
        print("\n=== Iteration ===")
        print("Starting training loop...")
        cmd, resp, train_response, env_resp, train_response_2 = ai.train_step("allocate", 4, 500)
        print(f"Command: {cmd}")
        print(f"Response: {resp}")
        print(f"Initial Training Feedback: {train_response}")
        print(f"Environmental Feedback: {env_resp}")
        print(f"Follow-up Training Feedback: {train_response_2}")
        print("---")
        # No fixed wait time to avoid interrupting LM Studio
        print("Waiting for next iteration... (Use escape commands: 'quit', 'restart', 'stop_deepseek')")
        time.sleep(2)  # Short delay to prevent overwhelming the system
# EOF  # End of commented-out heredoc

# Step 5: Run the persistent training loop with escape command handling
echo "Starting persistent training loop..."
iteration=0
while true; do
    echo "=== Iteration $iteration ==="
    python3 ai_model.py
    echo "Completed iteration. Check lm_guidance.log for training feedback, lm_env.log for environmental updates, and tool_changes.log for tool modifications."
    echo "Escape commands: Send 'quit', 'restart', or 'stop_deepseek' to /home/raymond/aios_project/escape_fifo"
    # Check for escape commands via named pipe
    if [ -p "$FIFO" ]; then
        if read -t 2 command < "$FIFO"; then
            case $command in
                "quit")
                    echo "Received 'quit' command, stopping all processes..."
                    cleanup
                    ;;
                "restart")
                    echo "Received 'restart' command, restarting QEMU..."
                    if [ -f "$QEMU_PID_FILE" ]; then
                        QEMU_PID=$(cat "$QEMU_PID_FILE")
                        if ps -p $QEMU_PID > /dev/null 2>&1; then
                            echo "Stopping QEMU (PID: $QEMU_PID)..."
                            kill $QEMU_PID
                            sleep 1
                            if ps -p $QEMU_PID > /dev/null 2>&1; then
                                echo "QEMU did not stop gracefully, forcing termination..."
                                kill -9 $QEMU_PID
                            fi
                        fi
                        rm -f "$QEMU_PID_FILE"
                    fi
                    echo "Starting QEMU with updated AIOS core..."
                    qemu-system-x86_64 -drive file=aios.bin,format=raw -monitor tcp:localhost:4444,server,nowait -m 512M &
                    QEMU_PID=$!
                    sleep 2
                    if ! ps -p $QEMU_PID > /dev/null; then
                        echo "Failed to start QEMU"
                        exit 1
                    fi
                    echo $QEMU_PID > "$QEMU_PID_FILE"
                    echo "QEMU restarted with PID: $QEMU_PID"
                    ;;
                "stop_deepseek")
                    echo "Received 'stop_deepseek' command, stopping DeepSeek task..."
                    DEEPSEEK_RUNNING=0
                    if [ -n "$DEEPSEEK_PID" ] && ps -p $DEEPSEEK_PID > /dev/null 2>&1; then
                        echo "Stopping DeepSeek task (PID: $DEEPSEEK_PID)..."
                        kill $DEEPSEEK_PID
                        sleep 1
                        if ps -p $DEEPSEEK_PID > /dev/null 2>&1; then
                            echo "DeepSeek task did not stop gracefully, forcing termination..."
                            kill -9 $DEEPSEEK_PID
                        fi
                        DEEPSEEK_PID=""
                    fi
                    echo "DeepSeek task stopped."
                    ;;
            esac
        fi
    fi
    ((iteration++))
    sleep 2  # Reduced delay to 2 seconds
done
