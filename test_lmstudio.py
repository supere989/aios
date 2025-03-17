#!/usr/bin/env python3
"""
Test LM Studio Script

This script uses the code_editor.py utility to create a new Python file
called "green_hello.py" that prints "Hello World" in green text using ANSI color codes.
"""

import subprocess
import os
import sys

def main():
    """
    Main function to create a green Hello World Python script using code_editor.py
    """
    print("Creating a Python script that prints 'Hello World' in green text...")
    
    # Define the path to code_editor.py
    code_editor_path = "./code_editor.py"
    
    # Check if code_editor.py exists
    if not os.path.exists(code_editor_path):
        print("Error: code_editor.py not found in the current directory!")
        sys.exit(1)
    
    # Create the content for green_hello.py
    # This script will use ANSI color codes to print text in green
    green_hello_content = """#!/usr/bin/env python3
'''
Green Hello World Script

This script prints "Hello World" in green text using ANSI color codes.
'''

# ANSI color codes
# \033[32m sets the text color to green
# \033[0m resets the text color to default

def print_green_text(text):
    '''
    Function to print text in green color
    
    Args:
        text (str): The text to be printed in green
    '''
    # Set text color to green, print the text, then reset the color
    print(f"\\033[32m{text}\\033[0m")

if __name__ == "__main__":
    # Call the function to print "Hello World" in green
    print_green_text("Hello World")
    
    # Print a message explaining what happened
    print("The text above is printed in green using ANSI color code \\033[32m")
"""
    
    try:
        # Command to create the green_hello.py file using code_editor.py
        cmd = [
            code_editor_path,
            "create",
            "green_hello.py",
            "--content", green_hello_content,
            "--force"  # Overwrite if the file already exists
        ]
        
        # Execute the command
        print("Executing code_editor.py to create green_hello.py...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check if the command was successful
        if result.returncode == 0:
            print(f"Successfully created green_hello.py")
            print("\nYou can run the script with: python green_hello.py")
            
            # Make the script executable
            try:
                os.chmod("green_hello.py", 0o755)
                print("Made green_hello.py executable")
            except Exception as e:
                print(f"Warning: Could not make green_hello.py executable: {e}")
                
            # Display the contents of the file
            print("\nContents of green_hello.py:")
            display_cmd = [code_editor_path, "display", "green_hello.py", "--line-numbers"]
            display_result = subprocess.run(display_cmd, capture_output=True, text=True)
            print(display_result.stdout)
            
        else:
            print(f"Error creating green_hello.py: {result.stderr}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error executing code_editor.py: {e}")
        sys.exit(1)
    
if __name__ == "__main__":
    main()

