#!/usr/bin/env python3
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QLabel, QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
import logging

class ModelStateView(QWidget):
    """Widget for displaying the current state and operations of the AI Model"""
    
    stateUpdated = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
        # Title with enhanced styling
        self.title = QLabel("AI Model State and Operations")
        self.title.setStyleSheet("font-weight: bold; font-size: 14px; color: #2c3e50;")
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
        
        # Configure table properties with enhanced styling
        header = self.table.horizontalHeader()
        for i in range(5):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        
        self.table.setStyleSheet("""
            QTableWidget {
                gridline-color: #d0d0d0;
                background-color: #ffffff;
                border: 1px solid #c0c0c0;
                border-radius: 4px;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 4px;
                border: 1px solid #d0d0d0;
                font-weight: bold;
            }
            QTableWidget::item {
                padding: 4px;
            }
        """)
        
        self.table.verticalHeader().setVisible(False)
        self.layout.addWidget(self.table)
        
        # Current state information with styling
        self.state_info = QTextEdit()
        self.state_info.setReadOnly(True)
        self.state_info.setMaximumHeight(100)
        self.state_info.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 4px;
            }
        """)
        self.layout.addWidget(self.state_info)
        
        self.last_row = -1
        
    def add_operation(self, llm_task, assembly, qemu_cmd, qemu_response, feedback):
        """Add a new operation to the table"""
        self.last_row += 1
        self.table.insertRow(self.last_row)
        
        items = [
            self._create_table_item(llm_task),
            self._create_table_item(assembly),
            self._create_table_item(qemu_cmd),
            self._create_table_item(qemu_response, self._get_response_color(qemu_response)),
            self._create_table_item(feedback, self._get_feedback_color(feedback))
        ]
        
        for col, item in enumerate(items):
            self.table.setItem(self.last_row, col, item)
        
        # Adjust row height for better readability
        self.table.resizeRowToContents(self.last_row)
        self.table.scrollToBottom()
        self.stateUpdated.emit()
        
    def _get_response_color(self, response):
        """Determine color based on QEMU response"""
        response = response.lower()
        if "success" in response or "ok" in response:
            return QColor("#4caf50")  # Green for success
        elif "error" in response or "fail" in response:
            return QColor("#f44336")  # Red for error
        return QColor("#000000")  # Default black
        
    def _get_feedback_color(self, feedback):
        """Determine color based on feedback content"""
        feedback = feedback.lower()
        if "correct" in feedback or "good" in feedback:
            return QColor("#4caf50")  # Green for positive feedback
        elif "incorrect" in feedback or "error" in feedback:
            return QColor("#f44336")  # Red for negative feedback
        return QColor("#000000")  # Default black
        
    def update_state_info(self, memory_state, model_state):
        """Update the current state information with formatting"""
        info_text = (
            f"<b>Memory State:</b><br>{memory_state}<br><br>"
            f"<b>Model State:</b><br>{model_state}"
        )
        self.state_info.setHtml(info_text)
        self.stateUpdated.emit()
        
    def clear_history(self):
        """Clear the operation history"""
        self.table.setRowCount(0)
        self.last_row = -1
        self.state_info.clear()
        self.stateUpdated.emit()
        
    def _create_table_item(self, text, color=None):
        """Create a table item with the given text and color"""
        item = QTableWidgetItem(str(text))
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        if color is not None:
            item.setForeground(color)
        return item
        
    def export_history(self, filename):
        """Export the operation history to a file"""
        try:
            with open(filename, 'w') as f:
                headers = [self.table.horizontalHeaderItem(i).text() 
                         for i in range(self.table.columnCount())]
                f.write("\t".join(headers) + "\n")
                
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
