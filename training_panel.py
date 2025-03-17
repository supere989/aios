#!/usr/bin/env python3
"""
Training Panel Widget for AIOS

This module provides a PyQt6 widget for visualizing and managing AI model training states.
"""

import os
import json
import logging
import time
from datetime import datetime
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTextEdit, QLineEdit, QComboBox, QGroupBox, QFormLayout,
    QSpinBox, QTabWidget, QFileDialog, QMessageBox, QSplitter,
    QTableWidget, QTableWidgetItem, QCheckBox, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QTextCursor

# Import our AI controller
from ai_model import AIOSController, AIModel

class TrainingStateWidget(QWidget):
    """Widget for displaying AI model training state"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.controller = None
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI components"""
        layout = QVBoxLayout(self)
        
        # Current state display
        state_group = QGroupBox("Current Training State")
        state_layout = QFormLayout(state_group)
        
        self.model_label = QLabel("Not loaded")
        state_layout.addRow("Model:", self.model_label)
        
        self.status_label = QLabel("Idle")
        state_layout.addRow("Status:", self.status_label)
        
        self.iteration_label = QLabel("0")
        state_layout.addRow("Iteration:", self.iteration_label)
        
        self.reward_label = QLabel("0.0")
        state_layout.addRow("Reward:", self.reward_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        state_layout.addRow("Progress:", self.progress_bar)
        
        layout.addWidget(state_group)
        
        # Training log
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        log_layout.addWidget(self.log_display)
        
        layout.addWidget(log_group)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        
        self.save_button = QPushButton("Save Checkpoint")
        self.save_button.clicked.connect(self.save_checkpoint)
        
        self.load_button = QPushButton("Load Checkpoint")
        self.load_button.clicked.connect(self.load_checkpoint)
        
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.save_button)
        controls_layout.addWidget(self.load_button)
        layout.addLayout(controls_layout)
        
    def set_controller(self, controller):
        """Set the AI controller"""
        self.controller = controller
        self.model_label.setText(type(self.controller.model).__name__)
        
    def start_training(self):
        """Start AI model training"""
        if not self.controller:
            self.log_message("Error: No AI controller available")
            return
            
        self.log_message("Starting training...")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Training")
        
        # Here we would normally start a training thread
        # For now, we'll just simulate training with a timer
        self.training_timer = QTimer()
        self.training_timer.timeout.connect(self.training_step)
        self.training_timer.start(1000)  # Simulate a step every second
        
    def stop_training(self):
        """Stop AI model training"""
        if hasattr(self, 'training_timer'):
            self.training_timer.stop()
            
        self.log_message("Training stopped")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Idle")
        
    def training_step(self):
        """Simulate a training step (would be replaced with real training)"""
        if not self.controller:
            return
            
        # Increment iteration counter
        iteration = int(self.iteration_label.text())
        iteration += 1
        self.iteration_label.setText(str(iteration))
        
        # Simulate training by running one iteration of train_step
        try:
            cmd, resp, train1, env, train2 = self.controller.train_step("allocate", 4, 500)
            
            # Update reward display
            reward = "Positive" if "correct" in train1.lower() else "Negative" if "incorrect" in train1.lower() else "Neutral"
            self.reward_label.setText(reward)
            
            # Log the results
            self.log_message(f"Iteration {iteration}:")
            self.log_message(f"Command: {cmd}")
            self.log_message(f"Response: {resp}")
            self.log_message(f"Training feedback: {train1[:100]}...")
            self.log_message(f"Environment feedback: {env[:100]}...")
            self.log_message(f"Follow-up feedback: {train2[:100]}...")
            self.log_message("---")
            
        except Exception as e:
            self.log_message(f"Error in training step: {str(e)}")
            self.stop_training()
        
    def save_checkpoint(self):
        """Save a checkpoint of the current model state"""
        if not self.controller:
            self.log_message("Error: No AI controller available")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Checkpoint", "", 
            "PyTorch Models (*.pt);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            # Here we would normally save the model
            # For demonstration, we'll just save a simple file
            import torch
            torch.save(self.controller.model.state_dict(), file_path)
            
            self.log_message(f"Checkpoint saved to {file_path}")
        except Exception as e:
            self.log_message(f"Error saving checkpoint: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to save checkpoint: {str(e)}")
        
    def load_checkpoint(self):
        """Load a checkpoint into the model"""
        if not self.controller:
            self.log_message("Error: No AI controller available")
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Checkpoint", "", 
            "PyTorch Models (*.pt);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            # Here we would normally load the model
            # For demonstration, we'll just load a simple file
            import torch
            self.controller.model.load_state_dict(torch.load(file_path))
            
            self.log_message(f"Checkpoint loaded from {file_path}")
            self.model_label.setText(type(self.controller.model).__name__)
        except Exception as e:
            self.log_message(f"Error loading checkpoint: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load checkpoint: {str(e)}")
        
    def log_message(self, message):
        """Log a message to the display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_display.append(f"[{timestamp}] {message}")
        # Scroll to bottom
        cursor = self.log_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_display.setTextCursor(cursor)


class TrainingHistoryWidget(QWidget):
    """Widget for displaying training history"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = []  # List of training events
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI components"""
        layout = QVBoxLayout(self)
        
        # History table
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(["Timestamp", "Iteration", "Command", "Result"])
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.history_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.history_table.setAlternatingRowColors(True)
        layout.addWidget(self.history_table)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.clear_button = QPushButton("Clear History")
        self.clear_button.clicked.connect(self.clear_history)
        
        self.export_button = QPushButton("Export History")
        self.export_button.clicked.connect(self.export_history)
        
        controls_layout.addWidget(self.clear_button)
        controls_layout.addWidget(self.export_button)
        layout.addLayout(controls_layout)
        
    def add_history_item(self, iteration, command, result):
        """Add an item to the history table"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add to internal history list
        self.history.append({
            "timestamp": timestamp,
            "iteration": iteration,
            "command": command,
            "result": result
        })
        
        # Add to table
        row = self.history_table.rowCount()
        self.history_table.insertRow(row)
        
        self.history_table.setItem(row, 0, QTableWidgetItem(timestamp))
        self.history_table.setItem(row, 1, QTableWidgetItem(str(iteration)))
        self.history_table.setItem(row, 2, QTableWidgetItem(command))
        self.history_table.setItem(row, 3, QTableWidgetItem(result))
        
        # Auto-scroll to the new row
        self.history_table.scrollToItem(self.history_table.item(row, 0))
        
    def clear_history(self):
        """Clear the history table"""
        self.history = []
        self.history_table.setRowCount(0)
        
    def export_history(self):
        """Export the history to a JSON file"""
        if not self.history:
            QMessageBox.warning(self, "Warning", "No history to export")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export History", "", 
            "JSON Files (*.json);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'w') as f:
                json.dump(self.history, f, indent=2)
            
            QMessageBox.information(self, "Success", f"History exported to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export history: {str(e)}")


class TrainingPanel(QWidget):
    """Main Training Panel widget that integrates all training-related components"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.controller = AIOSController()
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI components"""
        layout = QVBoxLayout(self)
        
        # Tab widget for different panels
        self.tab_widget = QTabWidget()
        
        # Create state widget
        self.state_widget = TrainingStateWidget()
        self.state_widget.set_controller(self.controller)
        self.tab_widget.addTab(self.state_widget, "Current State")
        
        # Create history widget
        self.history_widget = TrainingHistoryWidget()
        self.tab_widget.addTab(self.history_widget, "Training History")
        
        # Add tab widget to layout
        layout.addWidget(self.tab_widget)
        
        # Connect signals
        # We would normally connect more signals here to update the history
        # For demonstration purposes, we'll update it in the training_step method
        
    def start_training(self):
        """Start training"""
        self.state_widget.start_training()
        
    def stop_training(self):
        """Stop training"""
        self.state_widget.stop_training()
        
    def save_checkpoint(self):
        """Save a checkpoint"""
        self.state_widget.save_checkpoint()
        
    def load_checkpoint(self):
        """Load a checkpoint"""
        self.state_widget.load_checkpoint()


if __name__ == "__main__":
    """Run the panel as a standalone application for testing"""
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    panel = TrainingPanel()
    panel.show()
    sys.exit(app.exec())

