#!/usr/bin/env python3
"""
QEMU Panel Widget for AIOS

This module provides a PyQt6 widget for controlling and visualizing QEMU VMs.
"""

import os
import json
import time
import threading
import logging
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTextEdit, QLineEdit, QComboBox, QGroupBox, QFormLayout,
    QSpinBox, QTabWidget, QFileDialog, QMessageBox, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QTextCursor

# Import our QEMU controller
from qemu_controller import QEMUController, QEMUConfig

class QEMUMonitorWidget(QWidget):
    """Widget for interacting with the QEMU monitor"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.controller = None
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI components"""
        layout = QVBoxLayout(self)
        
        # Monitor output display
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        self.output_display.setStyleSheet("background-color: #2D2D30; color: #FFFFFF; font-family: 'Courier New';")
        layout.addWidget(self.output_display)
        
        # Command input
        input_layout = QHBoxLayout()
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("Enter QEMU monitor command...")
        self.command_input.returnPressed.connect(self.send_command)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_command)
        
        input_layout.addWidget(self.command_input)
        input_layout.addWidget(self.send_button)
        layout.addLayout(input_layout)
        
        # Common commands dropdown
        common_layout = QHBoxLayout()
        common_layout.addWidget(QLabel("Common Commands:"))
        
        self.common_commands = QComboBox()
        self.common_commands.addItems([
            "info status", "info version", "info cpus", "info memory",
            "info block", "info network", "system_powerdown", "stop", "cont"
        ])
        
        self.quick_send = QPushButton("Quick Send")
        self.quick_send.clicked.connect(self.send_quick_command)
        
        common_layout.addWidget(self.common_commands)
        common_layout.addWidget(self.quick_send)
        layout.addLayout(common_layout)
        
    def set_controller(self, controller):
        """Set the QEMU controller"""
        self.controller = controller
        
    def send_command(self):
        """Send a command to the QEMU monitor"""
        if not self.controller:
            self.display_message("Error: No QEMU controller available")
            return
            
        command = self.command_input.text().strip()
        if not command:
            return
            
        self.display_message(f"> {command}")
        self.command_input.clear()
        
        if not self.controller.is_running():
            self.display_message("Error: VM is not running")
            return
            
        response = self.controller.send_monitor_command(command)
        self.display_message(response)
        
    def send_quick_command(self):
        """Send a command from the quick commands dropdown"""
        command = self.common_commands.currentText()
        self.command_input.setText(command)
        self.send_command()
        
    def display_message(self, message):
        """Display a message in the output"""
        self.output_display.append(message)
        # Scroll to bottom
        cursor = self.output_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.output_display.setTextCursor(cursor)
        
    def clear_display(self):
        """Clear the output display"""
        self.output_display.clear()

class QEMUConfigWidget(QWidget):
    """Widget for configuring QEMU VM settings"""
    
    config_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.controller = None
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI components"""
        layout = QVBoxLayout(self)
        
        # VM Configuration
        config_group = QGroupBox("VM Configuration")
        config_form = QFormLayout()
        
        # Memory size
        self.memory_size = QComboBox()
        self.memory_size.setEditable(True)
        self.memory_size.addItems(["128M", "256M", "512M", "1G", "2G", "4G"])
        config_form.addRow("Memory Size:", self.memory_size)
        
        # CPU cores
        self.cpu_cores = QSpinBox()
        self.cpu_cores.setRange(1, 8)
        self.cpu_cores.setValue(2)
        config_form.addRow("CPU Cores:", self.cpu_cores)
        
        # Disk image
        disk_layout = QHBoxLayout()
        self.disk_image = QLineEdit()
        self.disk_image.setReadOnly(True)
        
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_disk_image)
        
        disk_layout.addWidget(self.disk_image)
        disk_layout.addWidget(self.browse_button)
        config_form.addRow("Disk Image:", disk_layout)
        
        # Display mode
        self.display_mode = QComboBox()
        self.display_mode.addItems(["gtk", "sdl", "vnc", "nographic"])
        config_form.addRow("Display Mode:", self.display_mode)
        
        # KVM acceleration
        self.enable_kvm = QComboBox()
        self.enable_kvm.addItems(["Enabled", "Disabled"])
        config_form.addRow("KVM Acceleration:", self.enable_kvm)
        
        config_group.setLayout(config_form)
        layout.addWidget(config_group)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save Configuration")
        self.save_button.clicked.connect(self.save_config)
        
        self.create_disk_button = QPushButton("Create New Disk...")
        self.create_disk_button.clicked.connect(self.create_new_disk)
        
        buttons_layout.addWidget(self.save_button)
        buttons_layout.addWidget(self.create_disk_button)
        layout.addLayout(buttons_layout)
        
        # Add stretch to bottom to keep controls at top
        layout.addStretch()
        
    def set_controller(self, controller):
        """Set the QEMU controller and update UI"""
        self.controller = controller
        self.update_from_config()
        
    def update_from_config(self):
        """Update UI from controller config"""
        if not self.controller:
            return
            
        config = self.controller.config
        
        # Update UI elements
        self.memory_size.setCurrentText(config.memory_size)
        self.cpu_cores.setValue(config.cpu_cores)
        self.disk_image.setText(config.disk_image)
        self.display_mode.setCurrentText(config.display_mode)
        self.enable_kvm.setCurrentIndex(0 if config.enable_kvm else 1)
        
    def save_config(self):
        """Save the configuration to controller"""
        if not self.controller:
            QMessageBox.warning(self, "Error", "No QEMU controller available")
            return
            
        try:
            # Update controller config
            self.controller.config.memory_size = self.memory_size.currentText()
            self.controller.config.cpu_cores = self.cpu_cores.value()
            self.controller.config.disk_image = self.disk_image.text()
            self.controller.config.display_mode = self.display_mode.currentText()
            self.controller.config.enable_kvm = (self.enable_kvm.currentIndex() == 0)
            
            # Save to file
            self.controller.save_config()
            
            QMessageBox.information(self, "Success", "Configuration saved successfully")
            
            # Emit signal that config changed
            self.config_changed.emit()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save configuration: {str(e)}")
        
    def browse_disk_image(self):
        """Browse for a disk image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Disk Image", "", 
            "Disk Images (*.qcow2 *.img *.bin);;All Files (*)"
        )
        
        if file_path:
            self.disk_image.setText(file_path)
        
    def create_new_disk(self):
        """Create a new disk image"""
        if not self.controller:
            QMessageBox.warning(self, "Error", "No QEMU controller available")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Create Disk Image", "", 
            "QCOW2 Images (*.qcow2);;All Files (*)"
        )
        
        if not file_path:
            return
            
        # Add .qcow2 extension if not already present
        if not file_path.endswith(".qcow2"):
            file_path += ".qcow2"
            
        size, ok = QMessageBox.question(
            self, "Disk Size", 
            "What size would you like the disk to be?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Yes
        )
        
        sizes = {
            QMessageBox.StandardButton.Yes: "10G",  # Small (10GB)
            QMessageBox.StandardButton.No: "50G",   # Medium (50GB)
            QMessageBox.StandardButton.Cancel: "100G"  # Large (100GB)
        }
        
        if size in sizes:
            disk_size = sizes[size]
            
            try:
                if self.controller.create_disk_image(file_path, disk_size):
                    self.disk_image.setText(file_path)
                    QMessageBox.information(
                        self, "Success", 
                        f"Created disk image at {file_path} with size {disk_size}"
                    )
                else:
                    QMessageBox.warning(
                        self, "Error", 
                        f"Failed to create disk image"
                    )
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", 
                    f"Failed to create disk image: {str(e)}"
                )

class QEMUControlWidget(QWidget):
    """Widget for controlling QEMU VM (start, stop, pause, etc.)"""
    
    vm_started = pyqtSignal()
    vm_stopped = pyqtSignal()
    vm_status_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.controller = None
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI components"""
        layout = QVBoxLayout(self)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start VM")
        self.start_button.clicked.connect(self.start_vm)
        
        self.stop_button = QPushButton("Stop VM")
        self.stop_button.clicked.connect(self.stop_vm)
        self.stop_button.setEnabled(False)
        
        self.pause_button = QPushButton("Pause VM")
        self.pause_button.clicked.connect(self.pause_vm)
        self.pause_button.setEnabled(False)
        
        self.resume_button = QPushButton("Resume VM")
        self.resume_button.clicked.connect(self.resume_vm)
        self.resume_button.setEnabled(False)
        
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.resume_button)
        layout.addLayout(control_layout)
        
        # Status display
        status_group = QGroupBox("VM Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_display = QTextEdit()
        self.status_display.setReadOnly(True)
        self.status_display.setMaximumHeight(200)
        status_layout.addWidget(self.status_display)
        
        layout.addWidget(status_group)
        
    def set_controller(self, controller):
        """Set the QEMU controller"""
        self.controller = controller
        
    def start_vm(self):
        """Start the VM"""
        if not self.controller:
            self.status_display.append("Error: No QEMU controller available")
            return
            
        self.status_display.append("Starting VM...")
        if self.controller.start_vm():
            self.status_display.append("VM started successfully")
            self._update_button_states(running=True, paused=False)
            self.status_timer.start(1000)  # Update status every second
            self.vm_started.emit()
        else:
            self.status_display.append("Failed to start VM")
            
    def stop_vm(self):
        """Stop the VM"""
        if not self.controller:
            self.status_display.append("Error: No QEMU controller available")
            return
        
        self.status_display.append("Stopping VM...")
        if self.controller.stop_vm():
            self.status_display.append("VM stopped successfully")
            self._update_button_states(running=False, paused=False)
            self.status_timer.stop()
            self.vm_stopped.emit()
        else:
            self.status_display.append("Failed to stop VM")
            
    def pause_vm(self):
        """Pause the VM"""
        if not self.controller:
            self.status_display.append("Error: No QEMU controller available")
            return
            
        self.status_display.append("Pausing VM...")
        if self.controller.pause_vm():
            self.status_display.append("VM paused successfully")
            self._update_button_states(running=True, paused=True)
        else:
            self.status_display.append("Failed to pause VM")
            
    def resume_vm(self):
        """Resume the VM"""
        if not self.controller:
            self.status_display.append("Error: No QEMU controller available")
            return
            
        self.status_display.append("Resuming VM...")
        if self.controller.resume_vm():
            self.status_display.append("VM resumed successfully")
            self._update_button_states(running=True, paused=False)
        else:
            self.status_display.append("Failed to resume VM")
    
    def update_status(self):
        """Update the VM status display"""
        if not self.controller:
            return
            
        if not self.controller.is_running():
            self.status_display.append("VM is no longer running")
            self._update_button_states(running=False, paused=False)
            self.status_timer.stop()
            self.vm_stopped.emit()
            return
            
        # Get status from controller
        status = self.controller.get_vm_status()
        
        # Only update the display if the status changed
        if hasattr(self, 'last_status') and self.last_status == status:
            return
            
        self.last_status = status
        
        # Update buttons based on current status
        self._update_button_states(
            running=status.get("running", False),
            paused=status.get("paused", False)
        )
        
        # Emit signal with status
        self.vm_status_changed.emit(status)
    
    def _update_button_states(self, running, paused):
        """Update button enabled states based on VM status"""
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.pause_button.setEnabled(running and not paused)
        self.resume_button.setEnabled(running and paused)
        
    def clear_display(self):
        """Clear the status display"""
        self.status_display.clear()


class QEMUPanel(QWidget):
    """Main QEMU Panel widget that integrates all QEMU control components"""
    
    def __init__(self, parent=None, config_path="qemu_config.json"):
        super().__init__(parent)
        self.controller = QEMUController(config_path=config_path)
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI components"""
        layout = QVBoxLayout(self)
        
        # Tab widget for different panels
        self.tab_widget = QTabWidget()
        
        # Create control widget
        self.control_widget = QEMUControlWidget()
        self.control_widget.set_controller(self.controller)
        self.tab_widget.addTab(self.control_widget, "Control")
        
        # Create monitor widget
        self.monitor_widget = QEMUMonitorWidget()
        self.monitor_widget.set_controller(self.controller)
        self.tab_widget.addTab(self.monitor_widget, "Monitor")
        
        # Create config widget
        self.config_widget = QEMUConfigWidget()
        self.config_widget.set_controller(self.controller)
        self.tab_widget.addTab(self.config_widget, "Configuration")
        
        # Add tab widget to layout
        layout.addWidget(self.tab_widget)
        
        # Connect signals
        self.control_widget.vm_started.connect(self._on_vm_started)
        self.control_widget.vm_stopped.connect(self._on_vm_stopped)
        self.control_widget.vm_status_changed.connect(self._on_vm_status_changed)
        self.config_widget.config_changed.connect(self._on_config_changed)
        
    def _on_vm_started(self):
        """Handle VM started event"""
        self.monitor_widget.display_message("VM started")
        self.tab_widget.setCurrentWidget(self.monitor_widget)
        
    def _on_vm_stopped(self):
        """Handle VM stopped event"""
        self.monitor_widget.display_message("VM stopped")
        
    def _on_vm_status_changed(self, status):
        """Handle VM status changed event"""
        # Just a stub - this could be used to update a status bar or other widgets
        pass
        
    def _on_config_changed(self):
        """Handle configuration changed event"""
        def apply_config_changes():
            was_running = self.controller.is_running()
            
            if was_running:
                self.monitor_widget.display_message("Stopping VM to apply new configuration...")
                self.controller.stop_vm()
            
            # Allow time for VM to stop completely
            time.sleep(1)
            
            if was_running:
                self.monitor_widget.display_message("Restarting VM with new configuration...")
                if self.controller.start_vm():
                    self.monitor_widget.display_message("VM restarted successfully with new configuration")
                else:
                    self.monitor_widget.display_message("Failed to restart VM with new configuration")
            else:
                self.monitor_widget.display_message("Configuration updated successfully")
        
        # Run configuration changes in a separate thread
        config_thread = threading.Thread(target=apply_config_changes)
        config_thread.start()
        
    def start_vm(self):
        """Convenience method to start the VM"""
        self.control_widget.start_vm()
        
    def stop_vm(self):
        """Convenience method to stop the VM"""
        self.control_widget.stop_vm()
        
    def pause_vm(self):
        """Convenience method to pause the VM"""
        self.control_widget.pause_vm()
        
    def resume_vm(self):
        """Convenience method to resume the VM"""
        self.control_widget.resume_vm()
        
    def shutdown(self):
        """Properly shut down the QEMU VM if it's running"""
        if self.controller and self.controller.is_running():
            self.controller.stop_vm()


if __name__ == "__main__":
    """Run the panel as a standalone application for testing"""
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    panel = QEMUPanel()
    panel.show()
    sys.exit(app.exec())
