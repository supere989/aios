#!/usr/bin/env python3
"""
QEMU Controller Module for AIOS project
"""

import os
import json
import subprocess
import time
import logging
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

# Import the existing QEMUComm class
from qemu_comm import QEMUComm

@dataclass
class QEMUConfig:
    """Configuration for QEMU VM"""
    memory_size: str = "512M"
    disk_image: str = "aios.bin"
    display_mode: str = "gtk"
    cpu_cores: int = 2
    enable_kvm: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return vars(self)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QEMUConfig':
        return cls(**{k: v for k, v in data.items() if k in ["memory_size", "disk_image", "display_mode", "cpu_cores", "enable_kvm"]})

class QEMUController:
    """Controller for QEMU VM operations"""
    
    def __init__(self, config_path: str = "qemu_config.json"):
        self.logger = logging.getLogger("QEMUController")
        self.config_path = config_path
        self.config = QEMUConfig()
        self.qemu_process = None
        self.qemu_comm = None
        self.paused = False
        self.load_config()
        
    def load_config(self):
        """Load config from file if exists"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self.config = QEMUConfig.from_dict(json.load(f))
            except Exception as e:
                self.logger.error(f"Error loading config: {e}")
        else:
            self.save_config()
                
    def save_config(self):
        """Save config to file"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
            
    def start_vm(self) -> bool:
        """Start the QEMU VM"""
        if self.is_running():
            return True
            
        cmd = ["qemu-system-x86_64", "-m", self.config.memory_size]
        
        # Add disk image
        if os.path.exists(self.config.disk_image):
            if self.config.disk_image.endswith(".bin"):
                cmd.extend(["-drive", f"file={self.config.disk_image},format=raw"])
            else:
                cmd.extend(["-drive", f"file={self.config.disk_image},format=qcow2"])
                
        # Add display config
        if self.config.display_mode == "nographic":
            cmd.append("-nographic")
        else:
            cmd.extend(["-display", self.config.display_mode])
            
        # Add CPU configuration
        cmd.extend(["-smp", str(self.config.cpu_cores)])
        
        # Add KVM support if enabled
        if self.config.enable_kvm:
            cmd.append("-enable-kvm")
            
        # Add monitor interface
        cmd.extend(["-monitor", "tcp:localhost:4444,server,nowait"])
        
        try:
            self.qemu_process = subprocess.Popen(cmd)
            time.sleep(2)  # Wait for QEMU to start
            self._connect_monitor()
            return True
        except Exception as e:
            self.logger.error(f"Failed to start VM: {e}")
            return False
            
    def stop_vm(self, force=False) -> bool:
        """Stop the QEMU VM"""
        if not self.is_running():
            return True
            
        if not force and self.qemu_comm:
            self.qemu_comm.send_command("system_powerdown")
            for _ in range(5):  # Wait 5 seconds for clean shutdown
                if not self.is_running():
                    break
                time.sleep(1)
                
        if self.is_running():
            self.qemu_process.terminate()
            try:
                self.qemu_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.qemu_process.kill()
        
        self.qemu_process = None
        self.qemu_comm = None
        self.paused = False
        return True
        
    def pause_vm(self) -> bool:
        """Pause the QEMU VM"""
        if not self.is_running() or self.paused:
            return False
            
        if self.qemu_comm:
            self.qemu_comm.send_command("stop")
            self.paused = True
            return True
        return False
        
    def resume_vm(self) -> bool:
        """Resume the QEMU VM"""
        if not self.is_running() or not self.paused:
            return False
            
        if self.qemu_comm:
            self.qemu_comm.send_command("cont")
            self.paused = False
            return True
        return False
        
    def send_monitor_command(self, cmd: str) -> str:
        """Send command to QEMU monitor"""
        if not self.is_running() or not self.qemu_comm:
            return "Error: VM not running or monitor not connected"
            
        return self.qemu_comm.send_command(cmd)
        
    def is_running(self) -> bool:
        """Check if VM is running"""
        return self.qemu_process is not None and self.qemu_process.poll() is None
        
    def _connect_monitor(self) -> bool:
        """Connect to QEMU monitor"""
        try:
            self.qemu_comm = QEMUComm(host="localhost", port=4444)
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to monitor: {e}")
            return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    controller = QEMUController()
    print(f"Starting VM with config: {controller.config.to_dict()}")
    
    if controller.start_vm():
        print("VM started successfully")
        print(controller.send_monitor_command("info status"))
        
        print("Pausing VM...")
        controller.pause_vm()
        
        print("Resuming VM...")
        controller.resume_vm()
        
        input("Press Enter to stop VM...")
        controller.stop_vm()
    else:
        print("Failed to start VM")

