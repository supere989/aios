from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox,
    QLineEdit, QPushButton, QTextEdit, QLabel
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor

class ChatInterface(QWidget):
    """Chat interface widget with model selection and command input"""
    
    # Signals
    commandSubmitted = pyqtSignal(str, str)  # (model_name, command)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Chat history
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setMinimumHeight(150)
        self.chat_history.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        layout.addWidget(self.chat_history)
        
        # Input area
        input_layout = QHBoxLayout()
        
        # Model selector
        model_layout = QVBoxLayout()
        model_label = QLabel("Model:")
        model_label.setStyleSheet("font-weight: bold;")
        self.model_selector = QComboBox()
        self.model_selector.addItems([
            "LLM Model",
            "Assembly Translator",
            "QEMU Interface"
        ])
        self.model_selector.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: white;
                min-width: 150px;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 5px;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
        """)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_selector)
        input_layout.addLayout(model_layout)
        
        # Command input
        command_layout = QVBoxLayout()
        command_label = QLabel("Command:")
        command_label.setStyleSheet("font-weight: bold;")
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("Enter your command here...")
        self.command_input.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #66afe9;
            }
        """)
        self.command_input.returnPressed.connect(self.submit_command)
        command_layout.addWidget(command_label)
        command_layout.addWidget(self.command_input)
        input_layout.addLayout(command_layout)
        
        # Submit button
        self.submit_button = QPushButton("Send")
        self.submit_button.setStyleSheet("""
            QPushButton {
                padding: 5px 15px;
                background-color: #337ab7;
                color: white;
                border: none;
                border-radius: 4px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #286090;
            }
            QPushButton:pressed {
                background-color: #204d74;
            }
        """)
        self.submit_button.clicked.connect(self.submit_command)
        input_layout.addWidget(self.submit_button)
        
        layout.addLayout(input_layout)
        
    def submit_command(self):
        command = self.command_input.text().strip()
        if command:
            model = self.model_selector.currentText()
            self.add_to_history(f"You → {model}:", command)
            self.commandSubmitted.emit(model, command)
            self.command_input.clear()
        
    def add_to_history(self, sender, message):
        cursor = self.chat_history.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.chat_history.setTextCursor(cursor)
        
        # Format the message
        formatted_message = f"\n{sender}\n{message}\n"
        
        # Apply formatting
        cursor.insertHtml(f"<p><b>{sender}</b><br>{message}</p>")
        
        # Ensure the latest message is visible
        self.chat_history.verticalScrollBar().setValue(
            self.chat_history.verticalScrollBar().maximum()
        )
        
    def add_response(self, model, response):
        """Add a response from the model to the chat history"""
        self.add_to_history(f"{model} →", response)
        
    def clear_history(self):
        """Clear the chat history"""
        self.chat_history.clear()
