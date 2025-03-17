#!/usr/bin/env python3

import json
import os
from PyQt6.QtWidgets import (
    QDialog, QLabel, QLineEdit, QComboBox, QSpinBox, 
    QPushButton, QVBoxLayout, QHBoxLayout, QFormLayout,
    QDialogButtonBox, QGroupBox, QMessageBox
)
from PyQt6.QtCore import Qt, QSettings

class LLMConfigDialog(QDialog):
    """Dialog for configuring LLM connection settings."""
    
    def __init__(self, parent=None):
        """Initialize the dialog."""
        super().__init__(parent)
        self.setWindowTitle("LLM Configuration")
        self.setMinimumWidth(500)
        
        # Set up the UI
        self.setup_ui()
        
        # Load existing configuration if available
        self.load_config()

    def setup_ui(self):
        """Set up the user interface."""
        main_layout = QVBoxLayout(self)
        
        # Connection settings group
        conn_group = QGroupBox("Connection Settings")
        conn_layout = QFormLayout()
        
        # Primary URL
        self.primary_url = QLineEdit()
        self.primary_url.setPlaceholderText("http://localhost:4891/v1/chat/completions")
        conn_layout.addRow("Primary URL:", self.primary_url)
        
        # Backup URL
        self.backup_url = QLineEdit()
        self.backup_url.setPlaceholderText("http://localhost:4891/v1/chat/completions")
        conn_layout.addRow("Backup URL:", self.backup_url)
        
        # Timeout
        self.timeout = QSpinBox()
        self.timeout.setRange(5, 300)
        self.timeout.setValue(60)
        self.timeout.setSuffix(" seconds")
        conn_layout.addRow("Timeout:", self.timeout)
        
        conn_group.setLayout(conn_layout)
        main_layout.addWidget(conn_group)
        
        # Model settings group
        model_group = QGroupBox("Model Settings")
        model_layout = QFormLayout()
        
        # Training model
        self.training_model = QComboBox()
        self.training_model.setEditable(True)
        self.training_model.addItems([
            "deepseek-r1-distill-qwen-7b",
            "qwen2.5-7b-instruct-1m",
            "qwen2.5-coder-0.5b-instruct",
            "gemma-3-4b-it",
            "granite-3.2-8b-instruct"
        ])
        model_layout.addRow("Training Model:", self.training_model)
        
        # Environment model
        self.environment_model = QComboBox()
        self.environment_model.setEditable(True)
        self.environment_model.addItems([
            "qwen2.5-7b-instruct-1m",
            "deepseek-r1-distill-qwen-7b",
            "qwen2.5-coder-0.5b-instruct",
            "gemma-3-4b-it",
            "granite-3.2-8b-instruct"
        ])
        model_layout.addRow("Environment Model:", self.environment_model)
        
        model_group.setLayout(model_layout)
        main_layout.addWidget(model_group)
        
        # Refresh model list button
        refresh_button = QPushButton("Refresh Model List")
        refresh_button.clicked.connect(self.refresh_model_list)
        main_layout.addWidget(refresh_button)
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | 
                                       QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.save_config)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

    def refresh_model_list(self):
        """Fetch the list of available models from the LLM Studio server."""
        import requests
        
        url = self.primary_url.text()
        if not url:
            # If primary URL is empty, use the placeholder
            url = self.primary_url.placeholderText()
        
        # Extract base URL (remove the /chat/completions part)
        base_url = url.split('/v1/')[0] + '/v1/'
        models_url = base_url + 'models'
        
        try:
            response = requests.get(models_url, timeout=self.timeout.value())
            if response.status_code == 200:
                models_data = response.json()
                model_ids = [model['id'] for model in models_data['data']]
                
                # Update comboboxes while preserving current selection
                current_training = self.training_model.currentText()
                current_environment = self.environment_model.currentText()
                
                self.training_model.clear()
                self.environment_model.clear()
                
                self.training_model.addItems(model_ids)
                self.environment_model.addItems(model_ids)
                
                # Restore selections if they exist in the new list
                if current_training in model_ids:
                    self.training_model.setCurrentText(current_training)
                if current_environment in model_ids:
                    self.environment_model.setCurrentText(current_environment)
                
                QMessageBox.information(self, "Success", f"Found {len(model_ids)} models")
            else:
                QMessageBox.warning(self, "Error", f"Failed to get models: {response.status_code}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to connect to LLM Studio: {str(e)}")

    def get_config_path(self):
        """Get the path to the configuration file."""
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_config.json")

    def load_config(self):
        """Load configuration from file if it exists."""
        config_path = self.get_config_path()
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Set values in UI
                if 'primary_url' in config:
                    self.primary_url.setText(config['primary_url'])
                if 'backup_url' in config:
                    self.backup_url.setText(config['backup_url'])
                if 'timeout' in config:
                    self.timeout.setValue(config['timeout'])
                if 'training_model' in config:
                    self.training_model.setCurrentText(config['training_model'])
                if 'environment_model' in config:
                    self.environment_model.setCurrentText(config['environment_model'])
            except Exception as e:
                QMessageBox.warning(self, "Configuration Error", 
                                    f"Failed to load configuration: {str(e)}")

    def save_config(self):
        """Save the configuration to a JSON file and close the dialog."""
        config = {
            'primary_url': self.primary_url.text(),
            'backup_url': self.backup_url.text(),
            'timeout': self.timeout.value(),
            'training_model': self.training_model.currentText(),
            'environment_model': self.environment_model.currentText()
        }
        
        try:
            with open(self.get_config_path(), 'w') as f:
                json.dump(config, f, indent=4)
            
            QMessageBox.information(self, "Success", "Configuration saved successfully")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save configuration: {str(e)}")


if __name__ == "__main__":
    """Run the dialog as a standalone application for testing."""
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    dialog = LLMConfigDialog()
    dialog.exec()

