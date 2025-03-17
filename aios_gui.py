import sys
from PyQt6.QtWidgets import (
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QSplitter,
    QTreeWidget, QTreeWidgetItem, QTabWidget, QStatusBar,
    QMessageBox, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QTextCursor, QFont, QColor, QPalette
import logging
import json
def __init__(self):
    super().__init__()
    self.ai_controller = AIOSController()
    self.setup_ui()
    
    # Connect model state view to AI controller
        self.ai_controller.state_view = self.model_state_view

def connect_model_signals(self):
    """Connect model-related signals"""
    # Connect training signals
    self.training_panel.startTraining.connect(self.start_training)
    self.training_panel.stopTraining.connect(self.stop_training)
    self.qemu_panel.qemuStateChanged.connect(self.handle_qemu_state_change)
    
    # Connect model update signals from AI controller
    self.ai_controller.model_updated.connect(self.update_model_state)
    
def start_training(self):
    """Start the training process"""
    try:
        params = self.training_panel.get_training_params()
        self.model_state_view.update_state_info(
            memory_state="Starting training...",
            model_state="Initializing..."
        )
        self.ai_controller.start_training(params)
        self.statusBar.showMessage("Training started")
    except Exception as e:
        logging.error(f"Error starting training: {e}")
        self.show_error_message(str(e))
        
def stop_training(self):
    """Stop the training process"""
    try:
        self.ai_controller.stop_training()
        self.model_state_view.update_state_info(
            memory_state="Training stopped",
            model_state="Idle"
        )
        self.statusBar.showMessage("Training stopped")
    except Exception as e:
        logging.error(f"Error stopping training: {e}")
        self.show_error_message(str(e))
        
def handle_qemu_state_change(self, state):
    """Handle QEMU state changes"""
    try:
        if state == "running":
            self.model_state_view.update_state_info(
                memory_state="QEMU running",
                model_state=self.ai_controller._get_model_state()
            )
            self.statusBar.showMessage("QEMU running")
        elif state == "stopped":
            self.model_state_view.update_state_info(
                memory_state="QEMU stopped",
                model_state="Idle"
            )
            self.statusBar.showMessage("QEMU stopped")
    except Exception as e:
        logging.error(f"Error handling QEMU state change: {e}")
        self.show_error_message(str(e))

def update_model_state(self, state_info):
    """Update the model state view with new information"""
    try:
        if 'operation' in state_info:
            self.model_state_view.add_operation(
                llm_task=state_info.get('llm_task', ''),
                assembly=state_info.get('assembly', ''),
                qemu_cmd=state_info.get('qemu_cmd', ''),
                qemu_response=state_info.get('qemu_response', ''),
                feedback=state_info.get('feedback', '')
            )
        
        if 'memory_state' in state_info or 'model_state' in state_info:
            self.model_state_view.update_state_info(
                memory_state=state_info.get('memory_state', ''),
                model_state=state_info.get('model_state', '')
            )
        
        self.statusBar.showMessage("Model state updated")
    except Exception as e:
        logging.error(f"Error updating model state: {e}")
        self.show_error_message(str(e))

def show_error_message(self, message):
    """Show error message in status bar and dialog"""
    self.statusBar.showMessage(f"Error: {message}")
    QMessageBox.critical(self, "Error", message)

def create_menu_bar(self):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Title
        self.title = QLabel("AI Model State and Operations")
        self.layout.addWidget(self.title)
        
        # Create table for state display
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([
            "LLM Task Input",
            "Assembly Code",
            "QEMU Command",
            "QEMU Response",
            "Learning Feedback"
        ])
        
        # Configure table properties
        header = self.table.horizontalHeader()
        for i in range(5):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        
        self.table.verticalHeader().setVisible(False)
        self.layout.addWidget(self.table)
        
        # Current state information
        self.state_info = QTextEdit()
        self.state_info.setReadOnly(True)
        self.state_info.setMaximumHeight(100)
        self.layout.addWidget(self.state_info)
        
        self.last_row = -1
        
    def add_operation(self, llm_task, assembly, qemu_cmd, qemu_response, feedback):
        """Add a new operation to the table"""
        self.last_row += 1
        self.table.insertRow(self.last_row)
        
        # Create and set items
        items = [
            self._create_table_item(llm_task),
            self._create_table_item(assembly),
            self._create_table_item(qemu_cmd),
            self._create_table_item(qemu_response),
            self._create_table_item(feedback)
        ]
        
        for col, item in enumerate(items):
            self.table.setItem(self.last_row, col, item)
        
        # Scroll to the new row
        self.table.scrollToBottom()
        self.stateUpdated.emit()
        
    def update_state_info(self, memory_state, model_state):
        """Update the current state information"""
        info_text = f"Memory State:\n{memory_state}\n\nModel State:\n{model_state}"
        self.state_info.setText(info_text)
        self.stateUpdated.emit()
        
    def clear_history(self):
        """Clear the operation history"""
        self.table.setRowCount(0)
        self.last_row = -1
        self.state_info.clear()
        self.stateUpdated.emit()
        
    def _create_table_item(self, text):
        """Create a table item with the given text"""
        item = QTableWidgetItem(str(text))
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)  # Make read-only
        return item
        
    def export_history(self, filename):
        """Export the operation history to a file"""
        try:
            with open(filename, 'w') as f:
                # Write headers
                headers = [self.table.horizontalHeaderItem(i).text() 
                         for i in range(self.table.columnCount())]
                f.write("\t".join(headers) + "\n")
                
                # Write data
                for row in range(self.table.rowCount()):
                    row_data = []
                    for col in range(self.table.columnCount()):
                        item = self.table.item(row, col)
                        row_data.append(item.text() if item else "")
                    f.write("\t".join(row_data) + "\n")
            return True
        except Exception as e:
            logging.error(f"Error exporting history: {e}")
            return False
import subprocess
import tempfile
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QSplitter, QWidget, QVBoxLayout, QHBoxLayout,
    QPlainTextEdit, QTextEdit, QPushButton, QLabel, QLineEdit, QListWidget,
    QMenu, QMenuBar, QStatusBar, QFileDialog, QMessageBox, QTabWidget,
    QComboBox, QToolBar, QSizePolicy, QCompleter, QListWidgetItem, QDialog
)
from PyQt6.QtCore import Qt, QProcess, pyqtSignal, QStringListModel
from PyQt6.QtGui import QColor, QTextCharFormat, QSyntaxHighlighter, QFont, QAction

# Import custom modules
# Import custom modules
from code_editor import CodeEditor
from ai_model import AIOSController
from llm_config import LLMConfigDialog
from qemu_panel import QEMUPanel
from training_panel import TrainingPanel
from model_state_view import ModelStateView
import json
import os
class CodeEditorWidget(QPlainTextEdit):
    """
    Enhanced code editor widget with syntax highlighting and line numbering
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_editor()
        
    def setup_editor(self):
        """Set up the code editor with appropriate font and settings"""
        font = QFont("Courier New", 10)
        font.setFixedPitch(True)
        self.setFont(font)
        
        # Set tab size to 4 spaces
        metrics = self.fontMetrics()
        self.setTabStopDistance(4 * metrics.horizontalAdvance(' '))
        
        # Other settings
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setBackgroundVisible(True)

class TerminalWidget(QPlainTextEdit):
    """
    Widget for displaying terminal output with process execution capability
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_terminal()
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self.read_stdout)
        self.process.readyReadStandardError.connect(self.read_stderr)
        self.process.finished.connect(self.process_finished)
        
    def setup_terminal(self):
        """Setup terminal appearance"""
        font = QFont("Courier New", 10)
        font.setFixedPitch(True)
        self.setFont(font)
        self.setReadOnly(True)
        
        # Dark background for terminal
        self.setStyleSheet("background-color: #2D2D30; color: #FFFFFF;")
    
    def execute_command(self, command):
        """Execute a shell command and display output"""
        self.appendPlainText(f"\n$ {command}\n")
        self.process.start("bash", ["-c", command])
    
    def execute_python(self, code):
        """Execute Python code and display output"""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp:
            temp_name = temp.name
            temp.write(code.encode())
        
        self.appendPlainText(f"\n>>> Running Python code...\n")
        self.process.start("python", [temp_name])
    
    def read_stdout(self):
        """Read and display standard output from process"""
        data = self.process.readAllStandardOutput().data().decode()
        self.appendPlainText(data)
    
    def read_stderr(self):
        """Read and display standard error from process"""
        data = self.process.readAllStandardError().data().decode()
        self.appendPlainText(f"ERROR: {data}")
    
    def process_finished(self, exitCode, exitStatus):
        """Handle process completion"""
        if exitCode == 0:
            self.appendPlainText("\nProcess completed successfully.")
        else:
            self.appendPlainText(f"\nProcess failed with exit code {exitCode}.")

class ChatWidget(QWidget):
    """
    Widget for interacting with the LLM
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ai_controller = self.create_ai_controller()
        self.chat_history = []
        self.setup_ui()
        
    def create_ai_controller(self):
        """Create an AI controller with settings from llm_config.json if available"""
        controller = AIOSController()
        
        # Check for LLM configuration file
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Update controller settings
                if 'primary_url' in config and config['primary_url']:
                    controller.lm_url = config['primary_url']
                if 'backup_url' in config and config['backup_url']:
                    controller.lm_backup_url = config['backup_url']
                if 'timeout' in config:
                    controller.lm_timeout = config['timeout']
                
                print(f"Loaded LLM configuration from {config_path}")
            except Exception as e:
                print(f"Error loading LLM configuration: {str(e)}")
        
        return controller
        
    def setup_ui(self):
        """Setup the chat interface"""
        layout = QVBoxLayout(self)
        
        # Chat display area
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display)
        
        # Input area
        input_layout = QHBoxLayout()
        self.chat_input = QTextEdit()
        self.chat_input.setMaximumHeight(100)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.chat_input)
        input_layout.addWidget(self.send_button)
        layout.addLayout(input_layout)
    
    def send_message(self):
        """Send message to LLM and display response"""
        user_message = self.chat_input.toPlainText().strip()
        if not user_message:
            return
            
        # Display user message
        self.display_message("You", user_message)
        self.chat_input.clear()
        
        # Save to history
        self.chat_history.append({"role": "user", "content": user_message})
        
        # Get LLM response
        try:
            # Use AI controller to get response
            response = self.ai_controller.query_lm_studio(user_message)
            
            # Display AI response
            self.display_message("AI", response)
            
            # Save to history
            self.chat_history.append({"role": "assistant", "content": response})
            
        except Exception as e:
            self.display_message("System", f"Error: {str(e)}", error=True)
    
    def display_message(self, sender, message, error=False):
        """Display a message in the chat display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.chat_display.append(f"<b>[{timestamp}] {sender}:</b>")
        
        if error:
            self.chat_display.append(f"<span style='color:red;'>{message}</span>")
        else:
            # Format code blocks if present
            formatted_message = message
            # Simple code block detection and formatting
            if "```" in message:
                formatted_message = self.format_code_blocks(message)
            
            self.chat_display.append(f"{formatted_message}")
        
        self.chat_display.append("<br>")
    
    def format_code_blocks(self, message):
        """Format code blocks in the message with syntax highlighting"""
        parts = message.split("```")
        formatted = parts[0]
        
        for i in range(1, len(parts)):
            if i % 2 == 1:  # This is a code block
                code = parts[i]
                if code.startswith("python\n"):
                    code = code[7:]  # Remove 'python\n'
                formatted += f"<pre style='background-color:#f0f0f0; padding:10px; border-radius:5px;'>{code}</pre>"
            else:
                formatted += parts[i]
                
        return formatted

class ChatHistoryWidget(QListWidget):
    """
    Widget for displaying and selecting chat history
    """
    chat_selected = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.chat_sessions = {}  # Dictionary of chat sessions
        self.itemDoubleClicked.connect(self.on_item_selected)
    
    def add_session(self, session_id, title, messages):
        """Add a chat session to history"""
        if session_id not in self.chat_sessions:
            self.chat_sessions[session_id] = messages
            item = QListWidgetItem(title)
            item.setData(Qt.ItemDataRole.UserRole, session_id)
            self.addItem(item)
    
    def on_item_selected(self, item):
        """Handle selection of a chat session"""
        session_id = item.data(Qt.ItemDataRole.UserRole)
        if session_id in self.chat_sessions:
            self.chat_selected.emit(self.chat_sessions[session_id])

class MainWindow(QMainWindow):
    """
    Main application window
    """
    def __init__(self):
        super().__init__()
        self.code_editor_util = CodeEditor()
        self.ai_controller = AIOSController()
        self.setup_ui()
        
        # Connect model state view to AI controller
        self.ai_controller.state_view = self.model_state_view
        
    def setup_ui(self):
        """Setup the main UI components"""
        self.setWindowTitle("AIOS - AI Code Assistant")
        self.resize(1200, 800)
        
        # Main layout with splitters
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel for file/chat history
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Chat history
        left_layout.addWidget(QLabel("<b>Chat History</b>"))
        self.chat_history_widget = ChatHistoryWidget()
        left_layout.addWidget(self.chat_history_widget)
        
        # Connect chat history selection to update chat
        self.chat_history_widget.chat_selected.connect(self.load_chat_session)
        
        # Add some test sessions
        self.chat_history_widget.add_session("1", "Session 1 - Code Generation", [])
        self.chat_history_widget.add_session("2", "Session 2 - Debugging", [])
        
        main_splitter.addWidget(left_panel)
        
        # Right split for editor and terminal
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Code editor area
        editor_terminal_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Code editor
        self.code_editor = CodeEditorWidget()
        editor_terminal_splitter.addWidget(self.code_editor)
        
        # Terminal output
        self.terminal = TerminalWidget()
        editor_terminal_splitter.addWidget(self.terminal)
        
        # Set initial sizes
        editor_terminal_splitter.setSizes([400, 200])
        
        right_splitter.addWidget(editor_terminal_splitter)
        
        # Create tab widget for different tools
        tools_tabs = QTabWidget()
        
        # Chat interface
        self.chat_widget = ChatWidget()
        tools_tabs.addTab(self.chat_widget, "AI Chat")
        
        # QEMU Panel
        # QEMU Panel
        self.qemu_panel = QEMUPanel()
        tools_tabs.addTab(self.qemu_panel, "QEMU VM")
        
        # Training Panel
        self.training_panel = TrainingPanel()
        tools_tabs.addTab(self.training_panel, "AI Training")
        
        # Create and add Model State View
        self.model_state_view = ModelStateView()
        tools_tabs.addTab(self.model_state_view, "Model State")
        
        # Add tools tabs to right splitter
        right_splitter.addWidget(tools_tabs)
        
        # Set initial sizes
        right_splitter.setSizes([600, 200])
        
        main_splitter.addWidget(right_splitter)
        
        # Set the main splitter as central widget
        self.setCentralWidget(main_splitter)
        main_splitter.setSizes([200, 1000])
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")

    def create_menu_bar(self):
        """Create the application menu bar"""
        # File menu
        file_menu = self.menuBar().addMenu("&File")
        
        # New file action
        new_action = QAction("&New", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_file)
        file_menu.addAction(new_action)
        
        # Open file action
        open_action = QAction("&Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        # Save file action
        save_action = QAction("&Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = self.menuBar().addMenu("&Edit")
        
        # Cut action
        cut_action = QAction("Cu&t", self)
        cut_action.setShortcut("Ctrl+X")
        cut_action.triggered.connect(self.code_editor.cut)
        edit_menu.addAction(cut_action)
        
        # Copy action
        copy_action = QAction("&Copy", self)
        copy_action.setShortcut("Ctrl+C")
        copy_action.triggered.connect(self.code_editor.copy)
        edit_menu.addAction(copy_action)
        
        # Paste action
        paste_action = QAction("&Paste", self)
        paste_action.setShortcut("Ctrl+V")
        paste_action.triggered.connect(self.code_editor.paste)
        edit_menu.addAction(paste_action)
        
        # Code menu
        code_menu = self.menuBar().addMenu("&Code")
        
        # Run code action
        run_action = QAction("&Run", self)
        run_action.setShortcut("F5")
        run_action.triggered.connect(self.run_code)
        code_menu.addAction(run_action)
        
        # Settings menu
        settings_menu = self.menuBar().addMenu("&Settings")
        
        # LLM Configuration action
        llm_config_action = QAction("&LLM Configuration", self)
        llm_config_action.triggered.connect(self.open_llm_config)
        settings_menu.addAction(llm_config_action)
        
        # QEMU menu
        qemu_menu = self.menuBar().addMenu("&QEMU")
        
        # Start VM action
        start_vm_action = QAction("&Start VM", self)
        start_vm_action.setShortcut("Ctrl+Alt+S")
        start_vm_action.triggered.connect(self.qemu_panel.start_vm)
        qemu_menu.addAction(start_vm_action)
        
        # Stop VM action
        stop_vm_action = QAction("S&top VM", self)
        stop_vm_action.setShortcut("Ctrl+Alt+T")
        stop_vm_action.triggered.connect(self.qemu_panel.stop_vm)
        qemu_menu.addAction(stop_vm_action)
        
        # Pause VM action
        pause_vm_action = QAction("&Pause VM", self)
        pause_vm_action.setShortcut("Ctrl+Alt+P")
        pause_vm_action.triggered.connect(self.qemu_panel.pause_vm)
        qemu_menu.addAction(pause_vm_action)
        
        # Resume VM action
        resume_vm_action = QAction("&Resume VM", self)
        resume_vm_action.setShortcut("Ctrl+Alt+R")
        resume_vm_action.triggered.connect(self.qemu_panel.resume_vm)
        resume_vm_action.triggered.connect(self.qemu_panel.resume_vm)
        qemu_menu.addAction(resume_vm_action)
        
        # Training menu
        training_menu = self.menuBar().addMenu("&Training")
        
        # Start training action
        start_training_action = QAction("&Start Training", self)
        start_training_action.setShortcut("Ctrl+T")
        start_training_action.triggered.connect(self.training_panel.start_training)
        training_menu.addAction(start_training_action)
        
        # Stop training action
        stop_training_action = QAction("S&top Training", self)
        stop_training_action.setShortcut("Ctrl+Shift+T")
        stop_training_action.triggered.connect(self.training_panel.stop_training)
        training_menu.addAction(stop_training_action)
        
        training_menu.addSeparator()
        
        # Save checkpoint action
        save_checkpoint_action = QAction("Save &Checkpoint", self)
        save_checkpoint_action.triggered.connect(self.training_panel.save_checkpoint)
        training_menu.addAction(save_checkpoint_action)
        
        # Load checkpoint action
        load_checkpoint_action = QAction("&Load Checkpoint", self)
        load_checkpoint_action.triggered.connect(self.training_panel.load_checkpoint)
        training_menu.addAction(load_checkpoint_action)
        
        # Help menu
        help_menu = self.menuBar().addMenu("&Help")
        
        # About action
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def new_file(self):
        """Create a new file"""
        self.code_editor.clear()
        self.setWindowTitle("AIOS - New File")
        self.statusBar.showMessage("New file created")
    
    def open_file(self):
        """Open a file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", "Python Files (*.py);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    self.code_editor.setPlainText(file.read())
                self.setWindowTitle(f"AIOS - {os.path.basename(file_path)}")
                self.statusBar.showMessage(f"Opened {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not open file: {str(e)}")
    
    def save_file(self):
        """Save the current file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save File", "", "Python Files (*.py);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as file:
                    file.write(self.code_editor.toPlainText())
                self.setWindowTitle(f"AIOS - {os.path.basename(file_path)}")
                self.statusBar.showMessage(f"Saved {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save file: {str(e)}")
    
    def run_code(self):
        """Execute the current code in the editor"""
        code = self.code_editor.toPlainText()
        if not code.strip():
            self.statusBar.showMessage("No code to run")
            return
            
        try:
            self.terminal.execute_python(code)
            self.statusBar.showMessage("Running code...")
        except Exception as e:
            self.statusBar.showMessage(f"Error executing code: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to run code: {str(e)}")
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, 
            "About AIOS", 
            """<h1>AIOS - AI Code Assistant</h1>
            <p>Version 1.0</p>
            <p>A Python GUI application that integrates code editing, execution, and AI assistance.</p>
            <p>Features:</p>
            <ul>
                <li>Code editing with syntax highlighting</li>
                <li>Integrated terminal for code execution</li>
                <li>LLM chat interface for AI assistance</li>
                <li>Chat history management</li>
            </ul>
            """
        )
    
    def load_chat_session(self, messages):
        """Load a chat session from history"""
        if not messages:
            return
            
        # Clear current chat display
        self.chat_widget.chat_display.clear()
        
        # Update chat history
        self.chat_widget.chat_history = messages
        
        # Display all messages in the session
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "user":
                self.chat_widget.display_message("You", content)
            elif role == "assistant":
                self.chat_widget.display_message("AI", content)
            elif role == "system":
                self.chat_widget.display_message("System", content, error=True)
        
        self.statusBar.showMessage(f"Loaded chat session with {len(messages)} messages")
    
    def open_llm_config(self):
        """Open the LLM configuration dialog"""
        dialog = LLMConfigDialog(self)
        result = dialog.exec()
        
        if result == 1:  # Accepted is 1, Rejected is 0
            # Get the old controller settings
            old_chat_history = self.chat_widget.chat_history
            
            # Create a new controller with updated settings
            self.chat_widget.ai_controller = self.chat_widget.create_ai_controller()
            
            # Restore chat history
            self.chat_widget.chat_history = old_chat_history
            
            self.statusBar.showMessage("LLM configuration updated")

    def closeEvent(self, event):
        """Handle application close event"""
        # Shut down QEMU VM if running
        if hasattr(self, 'qemu_panel'):
            self.qemu_panel.shutdown()
            
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
