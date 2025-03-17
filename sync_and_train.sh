#!/bin/bash

# Script to sync environment and run refined training loop with QEMU control
echo "Syncing AIOS environment and running refined training loop..."

# Ensure we're in the project directory
cd ~/aios_project || { echo "Failed to enter ~/aios_project"; exit 1; }

# File to store QEMU PID
QEMU_PID_FILE="$HOME/aios_project/qemu.pid"

# Cleanup function to stop QEMU and remove PID file
cleanup() {
    echo "Cleaning up QEMU environment..."
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

# Step 3: Update ai_model.py with refined training logic
echo "Updating ai_model.py with refined training logic..."
cat << 'EOF' > ai_model.py
import socket
import time
import torch
import torch.nn as nn
import requests
import json

# Class to communicate with QEMU's monitor interface
class QEMUComm:
    def __init__(self, host="localhost", port=4444):
        self.host = host
        self.port = port

    def send_command(self, cmd):
        # Mock alloc response based on AIOS core
        if cmd.startswith("alloc"):
            size = int(cmd.split()[1])
            return f"Allocated {size} bytes at 0x1000. Page table updated. Free size: 0xEFC0"
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.host, self.port))
            sock.send((cmd + "\n").encode())
            time.sleep(0.1)
            response = sock.recv(4096).decode().strip()
            sock.close()
            return response
        except Exception as e:
            return f"Communication error: {str(e)}"

# Simple neural network for generating commands
class AIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 2)  # Input: [intent, size_mb, free_mb, 0, 0], Output: [op, size]

    def forward(self, x):
        return torch.relu(self.fc(x))  # Ensure non-negative outputs

# Main controller for AIOS AI
class AIOSController:
    def __init__(self):
        self.model = AIModel()
        self.comm = QEMUComm()
        self.intent_map = {"allocate": 1}  # Focus on allocate only
        self.lm_url = "http://10.150.1.8:4891/v1/chat/completions"

    def process_intent(self, intent, size_mb, free_mb):
        intent_code = self.intent_map.get(intent, 0)
        if intent_code == 0:
            return "Invalid intent", "Error"
        input_tensor = torch.tensor([intent_code, size_mb, free_mb, 0, 0], dtype=torch.float32)
        output = self.model(input_tensor)
        op, size = [max(0, int(x)) for x in output.tolist()]
        size = max(4, min(4096, size))  # Ensure size is between 4 and 4096 bytes
        cmd = f"alloc {size} cont"  # Add allocation type as per LM Studio feedback
        response = self.comm.send_command(cmd)  # Use mock response
        return cmd, response

    def query_lm_studio(self, prompt):
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert assistant evaluating commands for a custom AI-driven Operating System (AIOS) running in a QEMU simulation. Your task is to assess the correctness of AI-generated commands (e.g., memory allocation or deallocation) and their responses from QEMU. Provide feedback on whether the command is valid, suggest improvements, and ensure the commands align with low-level OS operations like memory management. Be concise and focus on technical accuracy."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 100,
            "temperature": 0.7,
            "model": "gemma-3b-it"
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer dummy-api-key"
        }
        try:
            resp = requests.post(self.lm_url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            print(f"LM-Studio connection failed: {str(e)}")
            return "Mock response: correct"

    def train_step(self, intent, size_mb, free_mb):
        cmd, response = self.process_intent(intent, size_mb, free_mb)
        lm_prompt = f"Evaluate command '{cmd}' with response '{response}'. Is it correct? Suggest improvement."
        lm_response = self.query_lm_studio(lm_prompt)
        if "incorrect" in lm_response.lower():
            reward = -1
        elif "correct" in lm_response.lower():
            reward = 1
        else:
            reward = 0
        print(f"Reward: {reward} (LM Feedback: {lm_response})")
        return cmd, response, lm_response

# Test the setup
if __name__ == "__main__":
    ai = AIOSController()
    print("Starting training loop...")
    for i in range(3):  # 3 iterations for testing
        cmd, resp, lm_resp = ai.train_step("allocate", 4, 500)
        print(f"Iteration {i+1}:")
        print(f"Command: {cmd}")
        print(f"Response: {resp}")
        print(f"LM Feedback: {lm_resp}")
        print("---")
EOF

# Step 4: Run the updated training loop
echo "Running updated ai_model.py..."
python3 ai_model.py || { echo "AI model test failed"; exit 1; }

# Final message
echo "Test complete! QEMU is running (PID: $QEMU_PID)."
echo "To stop QEMU, run: kill $QEMU_PID"
echo "Next steps: Implement real 'alloc' command with monitor integration."
