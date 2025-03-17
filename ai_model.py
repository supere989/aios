import torch
import torch.nn as nn
import json
import logging
import requests
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class MemoryBlock:
    address: int
    size: int
    allocated: bool

class AIModel(nn.Module):
    def __init__(self, input_size: int = 5, hidden_size: int = 64, output_size: int = 4):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Enhanced encoder network
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU()
        )
        
        # Separate heads for different aspects of the command
        self.command_type_head = nn.Linear(self.hidden_size, 4)  # 4 command types
        self.param1_head = nn.Linear(self.hidden_size, 1)  # Continuous parameter
        self.param2_head = nn.Linear(self.hidden_size, 1)  # Continuous parameter
        self.size_head = nn.Linear(self.hidden_size, 1)  # Size parameter
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        return (
            self.command_type_head(encoded),  # Command type logits
            self.param1_head(encoded),        # Parameter 1
            self.param2_head(encoded),        # Parameter 2
            self.size_head(encoded)           # Size
        )

class AIOSController:
    def __init__(self):
        self.model = AIModel()
        self.memory_blocks: List[MemoryBlock] = []
        self.setup_logging()
        self.load_config()
        
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler('aios.log'),
                logging.StreamHandler()
            ]
        )
        
    def load_config(self):
        try:
            with open('llm_config.json', 'r') as f:
                config = json.load(f)
                self.primary_url = config.get('primary_url')
                self.backup_url = config.get('backup_url')
                self.timeout = config.get('timeout', 60)
                self.training_model = config.get('training_model')
                self.environment_model = config.get('environment_model')
        except FileNotFoundError:
            logging.info("Configuration file llm_config.json not found, using defaults")
            self.primary_url = "http://localhost:4891/v1/chat/completions"
            self.backup_url = self.primary_url
            self.timeout = 60
            self.training_model = "default-training-model"
            self.environment_model = "default-environment-model"

    def query_lm_studio(self, prompt: str, role: str, use_tools: bool = False) -> str:
        model = self.training_model if role == "training" else self.environment_model
        
        messages = [{"role": "user", "content": prompt}]
        
        # Add tool usage context if needed
        if use_tools:
            messages.insert(0, {
                "role": "system",
                "content": "You have access to tools for code modification and system analysis."
            })
        
        data = {
            "messages": messages,
            "model": model,
            "temperature": 0.7 if role == "training" else 0.3,
            "max_tokens": 1000
        }
        
        try:
            response = requests.post(
                self.primary_url,
                json=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            logging.error(f"Error querying LM Studio: {e}")
            if self.backup_url and self.backup_url != self.primary_url:
                try:
                    response = requests.post(
                        self.backup_url,
                        json=data,
                        timeout=self.timeout
                    )
                    response.raise_for_status()
                    return response.json()['choices'][0]['message']['content']
                except Exception as backup_e:
                    logging.error(f"Error querying backup LM Studio: {backup_e}")
            return f"Error: {str(e)}"

    def process_command(self, command_type: str, params: List[float]) -> Tuple[str, str]:
        # Convert command to tensor
        cmd_tensor = self._prepare_command_tensor(command_type, params)
        
        # Get model prediction
        with torch.no_grad():
            cmd_type_logits, param1, param2, size = self.model(cmd_tensor)
            
            # Convert logits to command ID
            cmd_id = torch.argmax(cmd_type_logits).item() + 1  # Add 1 since we start commands at 1
            
            # Get parameter values
            param1_val = param1.item()
            param2_val = param2.item()
            size_val = size.item()
        
        # Convert output to command
        cmd, response = self._execute_command(cmd_id, param1_val, param2_val, size_val)
        return cmd, response

    def _prepare_command_tensor(self, command_type: str, params: List[float]) -> torch.Tensor:
        command_codes = {
            "alloc": 1,
            "dealloc": 2,
            "meminfo": 3,
            "memcopy": 4
        }
        
        cmd_code = command_codes.get(command_type, 0)
        param1 = float(params[0]) if params else 0.0
        param2 = float(params[1]) if len(params) > 1 else 0.0
        free_mem = float(self._get_free_memory())
        used_mem = float(self._get_used_memory())
        
        return torch.tensor([cmd_code, param1, param2, free_mem, used_mem], dtype=torch.float32)

    def _execute_command(self, cmd_id: int, param1: float, param2: float, size: float) -> Tuple[str, str]:
        commands = {
            1: lambda: f"alloc {int(size)}",
            2: lambda: f"dealloc {int(param1)}",
            3: lambda: "meminfo",
            4: lambda: f"memcopy {int(param1)} {int(param2)}"
        }
        
        cmd = commands.get(cmd_id, lambda: "invalid")()
        response = self.comm.send_command(cmd)
        
        # Track memory operations
        if cmd_id == 1 and "success" in response.lower():  # alloc
            self.memory_blocks.append(MemoryBlock(int(param1), int(size), True))
        elif cmd_id == 2 and "success" in response.lower():  # dealloc
            for block in self.memory_blocks:
                if block.address == int(param1):
                    block.allocated = False
                    break
        
        return cmd, response

    def train_step(self, command_type: str, params: List[float]) -> Tuple[str, str, str, str, str, float]:
        cmd, response = self.process_command(command_type, params)
        
        # Get current memory state for context
        memory_state = self._get_memory_state()
        
        # Enhanced training prompt
        train_prompt = self._generate_training_prompt(cmd, response, command_type, params)
        train_response = self.query_lm_studio(train_prompt, "training", use_tools=False)
        
        # Enhanced environment feedback
        env_prompt = self._generate_environment_prompt(
            train_response, cmd, response, memory_state
        )
        env_resp = self.query_lm_studio(env_prompt, "environment", use_tools=True)
        
        # Final training evaluation
        train_prompt_2 = self._generate_final_evaluation_prompt(
            env_resp, cmd, response
        )
        train_response_2 = self.query_lm_studio(train_prompt_2, "training", use_tools=False)
        
        # Calculate reward
        reward = self._calculate_reward(
            cmd, response, 
            train_response, env_resp, train_response_2
        )
        
        return cmd, response, train_response, env_resp, train_response_2, reward

    def _generate_training_prompt(self, cmd: str, response: str, command_type: str, params: List[float]) -> str:
        return f"""
        Evaluate command '{cmd}' with response '{response}'.
        Command type: {command_type}
        Parameters: {params}
        
        Consider:
        1. Is the command syntax correct?
        2. Are the parameters appropriate?
        3. Is the memory operation valid?
        4. Did the operation succeed?
        
        Suggest specific improvements to:
        1. Command generation
        2. Parameter selection
        3. Memory management
        4. Error handling
        """

    def _generate_environment_prompt(self, train_response: str, cmd: str, response: str, memory_state: str) -> str:
        return f"""
        Trainer Feedback: {train_response}
        Command: {cmd}
        Response: {response}
        Memory State: {memory_state}
        
        Evaluate:
        1. Memory state consistency
        2. Command implementation
        3. Error handling
        4. Performance impact
        
        Suggest improvements to:
        1. Boot sector code
        2. Memory management
        3. Command processing
        4. Error recovery
        """

    def _generate_final_evaluation_prompt(self, env_resp: str, cmd: str, response: str) -> str:
        return f"""
        System Commander Feedback: {env_resp}
        Original Command: {cmd}
        Memory Operation Result: {response}
        
        Propose:
        1. Model architecture improvements
        2. Training data adjustments
        3. Parameter tuning
        4. New features needed
        """

    def _calculate_reward(self, cmd: str, response: str, train_resp: str, env_resp: str, train_resp_2: str) -> float:
        reward = 0.0
        
        # Command success/failure
        if "success" in response.lower():
            reward += 1.0
        if "error" in response.lower() or "fail" in response.lower():
            reward -= 1.0
            
        # Training feedback quality
        if "correct" in train_resp.lower():
            reward += 0.5
        if "incorrect" in train_resp.lower():
            reward -= 0.5
            
        # Environment feedback
        if "consistent" in env_resp.lower():
            reward += 0.5
        if "inconsistent" in env_resp.lower():
            reward -= 0.5
            
        # Learning progress
        if "improved" in train_resp_2.lower():
            reward += 0.5
            
        # Memory operation specific rewards
        cmd_parts = cmd.split()
        if cmd_parts[0] == "alloc":
            if all(block.address > 0 for block in self.memory_blocks if block.allocated):
                reward += 0.2  # Reward for maintaining valid memory addresses
        elif cmd_parts[0] == "dealloc":
            if any(not block.allocated for block in self.memory_blocks):
                reward += 0.2  # Reward for proper deallocation
                
        return reward

    def _get_memory_state(self) -> str:
        meminfo = self.comm.send_command("meminfo")
        return meminfo

    def _get_free_memory(self) -> int:
        meminfo = self._get_memory_state()
        try:
            # Parse the meminfo output to get free memory
            # This is a placeholder - implement actual parsing based on your meminfo format
            return int(meminfo.split("Free: ")[1].split()[0], 16)
        except Exception:
            return 0xF000  # Default to max if parsing fails

    def _get_used_memory(self) -> int:
        total = 0xF000  # Total memory size
        free = self._get_free_memory()
        return total - free

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
import pathlib
from requests.exceptions import RequestException, Timeout, ConnectionError

# Set default timeout for all requests
import requests.adapters
requests.adapters.DEFAULT_RETRIES = 1

# Define the MODIFY_FILE_TOOL for environment modifications
MODIFY_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "modify_file",
        "description": "Modifies the content of a file in the QEMU environment",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the file to be modified"
                },
                "content": {
                    "type": "string",
                    "description": "The new content to write to the file"
                },
                "operation": {
                    "type": "string",
                    "enum": ["append", "replace", "insert"],
                    "description": "The operation to perform on the file content"
                },
                "position": {
                    "type": "integer",
                    "description": "The position or line number where the content should be inserted or replaced, only needed for insert or replace operations"
                }
            },
            "required": ["file_path", "content", "operation"]
        }
    },
    "description": "Use this tool to modify files in the QEMU environment when system changes are needed."
}
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
        
        # Default configuration values
        self.lm_url = "http://10.150.1.8:4891/v1/chat/completions"
        self.lm_backup_url = "http://10.150.1.8:4891/v1/chat/completions"
        self.lm_timeout = 300  # Default 5 minutes timeout
        self.training_model = "deepseek-r1-distill-qwen-7b"
        self.environment_model = "qwen2.5-7b-instruct-1m"
        
        # Load configuration from llm_config.json if it exists
        self.load_config()
        
        # Override with environment variables if specified
        if os.environ.get("LM_STUDIO_URL"):
            self.lm_url = os.environ.get("LM_STUDIO_URL")
        if os.environ.get("LM_STUDIO_BACKUP_URL"):
            self.lm_backup_url = os.environ.get("LM_STUDIO_BACKUP_URL")
        
        self.lm_guidance_log = "lm_guidance.log"
        self.lm_env_log = "lm_env.log"
        self.tool_log = "tool_changes.log"
        self.logger.info(f"Using LM Studio URL: {self.lm_url} (backup: {self.lm_backup_url})")
        self.logger.info(f"Using models - Training: {self.training_model}, Environment: {self.environment_model}")
        # Initialize chat history for dual model interaction
        self.chat_history = [
            {"role": "system", "content": f"<start_of_turn>system\nThis is a collaborative chat between the Trainer ({self.training_model}) and System Commander ({self.environment_model}) to improve the AIOS system.<end_of_turn>\n"}
        ]

    def process_intent(self, intent, size_mb, free_mb):
        intent_code = self.intent_map.get(intent, 0)
        if intent_code == 0:
            return "Invalid intent", "Error"
        input_tensor = torch.tensor([intent_code, size_mb, free_mb, 0, 0], dtype=torch.float32)
        output = self.model(input_tensor)
        op, size = [max(0, int(x)) for x in output.tolist()]
        size = max(4, min(4096, size))  # Ensure size is between 4 and 4096 bytes
        
        # Handle the 'alloc' command which is implemented in boot.asm but not directly in QEMU
        # We'll send it to QEMU to be processed by our custom bootloader
        cmd = f"alloc {size}B"  # Add "B" for bytes
        response = self.comm.send_command(cmd)  # Use real QEMU response
        
        # Check if the command was not recognized by QEMU 
        if "unknown command" in response:
            self.logger.warning(f"Command '{cmd}' not recognized by QEMU monitor. Attempting alternative approach.")
            # Log this for analysis
            with open(self.tool_log, "a") as f:
                f.write(f"Command: {cmd}\nResponse: {response}\nAction: Attempting alternative approach\n---\n")
            
            # Try to use the info memory command to get system information
            memory_info = self.comm.send_command("info memory")
            self.logger.info(f"Memory info: {memory_info}")
            
            # Return the original command and response for learning purposes
        
        return cmd, response

    def load_config(self):
        """Load configuration from llm_config.json if it exists"""
        config_file = "llm_config.json"
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                self.logger.info(f"Loading configuration from {config_file}")
                
                # Update configuration values from file
                if 'primary_url' in config and config['primary_url']:
                    self.lm_url = config['primary_url']
                if 'backup_url' in config and config['backup_url']:
                    self.lm_backup_url = config['backup_url']
                if 'timeout' in config:
                    self.lm_timeout = config['timeout']
                if 'training_model' in config:
                    self.training_model = config['training_model']
                if 'environment_model' in config:
                    self.environment_model = config['environment_model']
                
                self.logger.info(f"Configuration loaded: {config}")
            except Exception as e:
                self.logger.error(f"Error loading configuration from {config_file}: {str(e)}")
        else:
            self.logger.info(f"Configuration file {config_file} not found, using defaults")

    def query_lm_studio(self, prompt, role="training", model=None, use_tools=False):
        # Update chat history with appropriate template
        if role == "training":
            formatted_prompt = "<start_of_turn>user\n" + prompt + "<end_of_turn>\n"
            self.chat_history.append({"role": "user", "content": formatted_prompt})
        else:
            formatted_prompt = prompt
            self.chat_history.append({"role": "user", "content": formatted_prompt})

        # Dual roles: Training Guidance and Environmental Updates
        system_content = ""
        if role == "training":
            system_content = f"<start_of_turn>system\nYou are the Trainer ({self.training_model}), an expert AI trainer improving the training logic for a custom AI-driven Operating System (AIOS) running in a QEMU simulation. Your task is to evaluate AI-generated commands and their responses, suggest improvements to the training process (e.g., reward system, model architecture), and provide Python code modifications to enhance the AI model in ai_model.py. Be concise and technically accurate.<end_of_turn>\n"
            model = model or self.training_model
            log_file = self.lm_guidance_log
            stop = ["<end_of_turn>"]
        elif role == "environment":
            model = model or self.environment_model
            log_file = self.lm_env_log
            stop = ["<end_of_turn>"]

        payload = {
            "messages": [{"role": "system", "content": system_content}] + self.chat_history if role == "training" else self.chat_history,
            "max_tokens": 200,
            "temperature": 0.7,
            "model": model,
            "stop": stop
        }
        # Enable tools for environment model (Qwen)
        if use_tools and role == "environment":
            payload["tools"] = [MODIFY_FILE_TOOL]
            payload["tool_choice"] = "auto"

        headers = {"Content-Type": "application/json", "Authorization": "Bearer lm-studio"}
        
        # Use the primary URL only
        url = self.lm_url
        
        # Fallback to backup URL only if primary is empty
        if not self.lm_url and self.lm_backup_url:
            url = self.lm_backup_url
            
        try:
            self.logger.info(f"Sending request to LM Studio at {url} with {self.lm_timeout}s timeout")
            self.logger.info(f"Waiting for complete model response (this may take up to 5 minutes)...")
            
            # Make a single request with a long timeout
            # Make a single request with the configured timeout
            resp = requests.post(url, json=payload, headers=headers, timeout=self.lm_timeout)
            
            response = resp.json()["choices"][0]["message"]
            content_to_log = response["content"].strip() if "content" in response and response["content"] else "No content in response."
            
            # Handle empty response with fallback text
            if not content_to_log or content_to_log == "No content in response.":
                if role == "training":
                    fallback_content = "The command appears to be formatted correctly, but the QEMU monitor doesn't support this command directly. Consider implementing this functionality at the OS level instead of relying on QEMU monitor commands."
                else:
                    fallback_content = "The QEMU monitor doesn't support this command. Consider modifying boot.asm to implement custom memory allocation routines that don't rely on QEMU monitor commands."
                
                self.logger.warning(f"Received empty response from model. Using fallback content.")
                content_to_log = fallback_content
            
            # Append response to chat history with proper formatting
            if role == "training":
                formatted_response = "<start_of_turn>model\n" + content_to_log + "<end_of_turn>\n"
            else:
                formatted_response = "<start_of_turn>assistant\n" + content_to_log + "<end_of_turn>\n"
            
            self.chat_history.append({"role": "assistant", "content": formatted_response})
            
            # Log to the appropriate file
            with open(log_file, "a") as f:
                f.write(f"Prompt: {prompt}\nFeedback: {content_to_log}\n---\n")
            
            self.logger.info(f"Successfully received complete response from {url}")
            return content_to_log
            
        except (ConnectionError, Timeout) as e:
            error_msg = f"Connection error to {url}: {str(e)}. No retry attempted to avoid breaking model generation."
            self.logger.error(error_msg)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Request error to {url}: {str(e)}. No retry attempted to avoid breaking model generation."
            self.logger.error(error_msg)
            
        except Exception as e:
            error_msg = f"Unexpected error when connecting to {url}: {str(e)}"
            self.logger.error(error_msg)
        
        # If we get here, something went wrong with the request
        with open(log_file, "a") as f:
            f.write(f"Prompt: {prompt}\nError: {error_msg}\n---\n")
            
        # Return appropriate fallback message based on role
        if role == "training":
            return "The command requires improvement. The QEMU monitor doesn't recognize this command. Consider implementing this functionality at the OS level or using valid QEMU monitor commands."
        else:
            return "Modify boot.asm to implement memory allocation that works properly with QEMU. The current approach is not recognized by the QEMU monitor."

    def train_step(self, command_type, params=None):
        """
        Execute a training step with the given command type and parameters.
        Args:
            command_type (str): Type of command to execute (alloc, dealloc, meminfo, memcopy)
            params (list): List of parameters for the command
        """
        if params is None:
            params = []

        # Get current memory state
        memory_info = self.comm.send_command("info memory")
        free_mb = 0xF000  # Default if can't parse memory info
        try:
            # Try to parse memory info
            if "Free: " in memory_info:
                free_mb = int(memory_info.split("Free: ")[1].split()[0], 16)
        except:
            pass

        # Process the command
        if command_type == "alloc":
            size_mb = params[0] if params else 4
            cmd, response = self.process_intent("allocate", size_mb, free_mb)
        else:
            cmd = f"{command_type} " + " ".join(map(str, params))
            response = self.comm.send_command(cmd)

        # Get current memory state for feedback
        memory_state = self.comm.send_command("meminfo")
        
        # Generate training prompts
        train_prompt = self._generate_training_prompt(cmd, response, command_type, params)
        train_response = self.query_lm_studio(train_prompt, "training", use_tools=False)
        
        env_prompt = self._generate_environment_prompt(train_response, cmd, response, memory_state)
        env_resp = self.query_lm_studio(env_prompt, "environment", use_tools=True)
        
        train_prompt_2 = self._generate_final_evaluation_prompt(env_resp, cmd, response)
        train_response_2 = self.query_lm_studio(train_prompt_2, "training", use_tools=False)
        
        # Calculate reward
        reward = self._calculate_reward(cmd, response, train_response, env_resp, train_response_2)
        
        return cmd, response, train_response, env_resp, train_response_2, reward

# Persistent loop to run the training and allow LM Studio models to collaborate
if __name__ == "__main__":
    ai = AIOSController()
    while True:
        print("\n=== Iteration ===")
        cmd, resp, train_response, env_resp, train_response_2, reward = ai.train_step("alloc", [4])
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
