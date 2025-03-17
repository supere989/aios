#!/usr/bin/env python3
"""
Code Editor Utility Script

This script provides utilities for code editing tasks including:
- Listing files in a directory with filtering options
- Searching for patterns in files (grep functionality)
- Applying changes to files using diff-like syntax
- Creating new files with code blocks
- Displaying file contents with optional line numbers

Usage:
    python code_editor.py <command> [options]

Commands:
    list      - List files in a directory with filtering options
    search    - Search for patterns in files
    diff      - Apply changes to files using diff-like syntax
    create    - Create a new file with code content
    display   - Display file contents with optional line numbers
    help      - Show help information for a specific command
"""

import os
import re
import sys
import glob
import argparse
import difflib
from typing import List, Optional, Dict, Tuple, Any
import fnmatch
import colorama
from colorama import Fore, Style

# Initialize colorama for cross-platform colored terminal output
colorama.init()


class CodeEditor:
    """Main class that handles code editing operations."""

    def __init__(self):
        """Initialize the CodeEditor."""
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """
        Create the argument parser for the command-line interface.

        Returns:
            argparse.ArgumentParser: The configured argument parser.
        """
        parser = argparse.ArgumentParser(
            description="Utility for code editing tasks",
            usage="code_editor.py <command> [options]"
        )
        
        # Add global verbose flag
        parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
        
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")
        
        # Create a parent parser with common arguments to share with all subparsers
        parent_parser = argparse.ArgumentParser(add_help=False)
        parent_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
        
        # List command
        list_parser = subparsers.add_parser("list", help="List files in a directory", parents=[parent_parser])
        list_parser.add_argument("directory", nargs="?", default=".", help="Directory to list (default: current directory)")
        list_parser.add_argument("--pattern", "-p", help="File pattern to match (e.g., '*.py')")
        list_parser.add_argument("--recursive", "-r", action="store_true", help="List files recursively")
        list_parser.add_argument("--long", "-l", action="store_true", help="Show detailed file information")
        
        # Search command
        search_parser = subparsers.add_parser("search", help="Search for patterns in files", parents=[parent_parser])
        search_parser.add_argument("pattern", help="Pattern to search for")
        search_parser.add_argument("files", nargs="+", help="Files to search in (wildcards supported)")
        search_parser.add_argument("--ignore-case", "-i", action="store_true", help="Ignore case in pattern matching")
        search_parser.add_argument("--whole-word", "-w", action="store_true", help="Match whole words only")
        search_parser.add_argument("--line-numbers", "-n", action="store_true", help="Show line numbers")
        
        # Diff command
        diff_parser = subparsers.add_parser("diff", help="Apply changes to files using diff-like syntax", parents=[parent_parser])
        diff_parser.add_argument("file", help="File to modify")
        diff_parser.add_argument("--search", "-s", required=True, help="Text to search for (multiline strings supported)")
        diff_parser.add_argument("--replace", "-r", required=True, help="Text to replace with (multiline strings supported)")
        diff_parser.add_argument("--preview", "-p", action="store_true", help="Preview changes without modifying the file")
        
        # Create command
        create_parser = subparsers.add_parser("create", help="Create a new file with code content", parents=[parent_parser])
        create_parser.add_argument("file", help="File to create")
        create_parser.add_argument("--content", "-c", help="Content to write to file")
        create_parser.add_argument("--from-stdin", "-s", action="store_true", help="Read content from standard input")
        create_parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing file")
        
        # Display command
        display_parser = subparsers.add_parser("display", help="Display file contents with optional line numbers", parents=[parent_parser])
        display_parser.add_argument("file", help="File to display")
        display_parser.add_argument("--line-numbers", "-n", action="store_true", help="Show line numbers")
        display_parser.add_argument("--range", "-r", help="Range of lines to display (e.g., '10-20')")
        display_parser.add_argument("--highlight", "-H", help="Pattern to highlight in the output")
        
        return parser

    def run(self) -> int:
        """
        Run the appropriate command based on command-line arguments.

        Returns:
            int: Exit code (0 for success, non-zero for errors)
        """
        args = self.parser.parse_args()
        
        if args.command is None:
            self.parser.print_help()
            return 0
        try:
            # Execute the appropriate command
            if args.command == "list":
                return self.list_files(args)
            elif args.command == "search":
                return self.search_files(args)
            elif args.command == "diff":
                return self.apply_diff(args)
            elif args.command == "create":
                return self.create_file(args)
            elif args.command == "display":
                return self.display_file(args)
            elif args.command == "help":
                self.parser.print_help()
                return 0
            else:
                print(f"Unknown command: {args.command}")
                return 1
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
            return 1

    def list_files(self, args: argparse.Namespace) -> int:
        """
        List files in a directory with optional filtering.

        Args:
            args: Command-line arguments

        Returns:
            int: Exit code (0 for success, non-zero for errors)
        """
        directory = args.directory
        pattern = args.pattern
        recursive = args.recursive
        long_format = args.long
        verbose = getattr(args, 'verbose', False)
        
        if verbose:
            print(f"{Fore.CYAN}========== LIST FILES =========={Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Directory:{Style.RESET_ALL} {directory}")
            print(f"{Fore.YELLOW}Pattern:{Style.RESET_ALL} {pattern if pattern else 'None'}")
            print(f"{Fore.YELLOW}Recursive:{Style.RESET_ALL} {recursive}")
            print(f"{Fore.YELLOW}Long format:{Style.RESET_ALL} {long_format}")
            print(f"{Fore.CYAN}============================={Style.RESET_ALL}")
        if not os.path.exists(directory):
            print(f"{Fore.RED}Error: Directory '{directory}' does not exist{Style.RESET_ALL}")
            return 1
        
        if not os.path.isdir(directory):
            print(f"{Fore.RED}Error: '{directory}' is not a directory{Style.RESET_ALL}")
            return 1
        
        files = []
        if verbose:
            print(f"\n{Fore.YELLOW}Scanning for files...{Style.RESET_ALL}")
            
        if recursive:
            if verbose:
                print(f"{Fore.CYAN}Scanning recursively in {directory}{Style.RESET_ALL}")
            for root, _, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(root, filename)
                    if pattern and not fnmatch.fnmatch(filepath, pattern):
                        continue
                    files.append(filepath)
                    if verbose and len(files) % 10 == 0:
                        print(f"{Fore.GREEN}Found {len(files)} files so far...{Style.RESET_ALL}")
        else:
            if verbose:
                print(f"{Fore.CYAN}Scanning directory {directory}{Style.RESET_ALL}")
            for item in os.listdir(directory):
                filepath = os.path.join(directory, item)
                if os.path.isfile(filepath):
                    if pattern and not fnmatch.fnmatch(filepath, pattern):
                        continue
                    files.append(filepath)
        if not files:
            msg = f"No files found in '{directory}'" + (f" matching '{pattern}'" if pattern else "")
            if verbose:
                print(f"{Fore.YELLOW}============================={Style.RESET_ALL}")
                print(f"{Fore.YELLOW}{msg}{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}============================={Style.RESET_ALL}")
            else:
                print(msg)
        
        # Sort files for consistent output
        files.sort()
        
        if verbose:
            print(f"\n{Fore.GREEN}Total: {len(files)} files to search{Style.RESET_ALL}")
            print(f"{Fore.CYAN}================================{Style.RESET_ALL}")
        if verbose:
            print(f"\n{Fore.GREEN}Found {len(files)} files.{Style.RESET_ALL}")
            print(f"{Fore.CYAN}============================={Style.RESET_ALL}")
        for file in files:
            if long_format:
                file_stat = os.stat(file)
                file_size = file_stat.st_size
                file_mode = file_stat.st_mode
                file_mtime = file_stat.st_mtime
                
                # Format file size
                if file_size < 1024:
                    size_str = f"{file_size} B"
                elif file_size < 1024 * 1024:
                    size_str = f"{file_size / 1024:.1f} KB"
                else:
                    size_str = f"{file_size / (1024 * 1024):.1f} MB"
                
                # Format file permissions (simplified)
                mode_str = ""
                mode_str += "r" if file_mode & 0o400 else "-"
                mode_str += "w" if file_mode & 0o200 else "-"
                mode_str += "x" if file_mode & 0o100 else "-"
                mode_str += "r" if file_mode & 0o040 else "-"
                mode_str += "w" if file_mode & 0o020 else "-"
                mode_str += "x" if file_mode & 0o010 else "-"
                mode_str += "r" if file_mode & 0o004 else "-"
                mode_str += "w" if file_mode & 0o002 else "-"
                mode_str += "x" if file_mode & 0o001 else "-"
                
                # Format modification time
                import datetime
                mtime_str = datetime.datetime.fromtimestamp(file_mtime).strftime("%Y-%m-%d %H:%M:%S")
                
                print(f"{mode_str}  {size_str:>8}  {mtime_str}  {file}")
            else:
                print(file)
        
        return 0

    def search_files(self, args: argparse.Namespace) -> int:
        """
        Search for patterns in files.

        Args:
            args: Command-line arguments

        Returns:
            int: Exit code (0 for success, non-zero for errors)
        """
        pattern = args.pattern
        ignore_case = args.ignore_case
        whole_word = args.whole_word
        line_numbers = args.line_numbers
        verbose = getattr(args, 'verbose', False)
        
        if verbose:
            print(f"{Fore.CYAN}========== SEARCH FILES =========={Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Pattern:{Style.RESET_ALL} {pattern}")
            print(f"{Fore.YELLOW}Ignore case:{Style.RESET_ALL} {ignore_case}")
            print(f"{Fore.YELLOW}Whole word:{Style.RESET_ALL} {whole_word}")
            print(f"{Fore.YELLOW}Show line numbers:{Style.RESET_ALL} {line_numbers}")
            print(f"{Fore.CYAN}================================{Style.RESET_ALL}")
        # Prepare pattern for whole word matching if needed
        if whole_word:
            pattern = r'\b' + re.escape(pattern) + r'\b'
        else:
            pattern = re.escape(pattern)
        
        # Compile regex
        flags = re.IGNORECASE if ignore_case else 0
        regex = re.compile(pattern, flags)
        
        # Expand file patterns and filter to existing files
        file_patterns = args.files
        files = []
        
        if verbose:
            print(f"\n{Fore.YELLOW}Expanding file patterns...{Style.RESET_ALL}")
            
        for file_pattern in file_patterns:
            if verbose:
                print(f"{Fore.CYAN}Processing pattern: {file_pattern}{Style.RESET_ALL}")
            matched_files = glob.glob(file_pattern, recursive=True)
            file_count = len([f for f in matched_files if os.path.isfile(f)])
            files.extend([f for f in matched_files if os.path.isfile(f)])
            if verbose:
                print(f"{Fore.GREEN}  Found {file_count} files matching pattern{Style.RESET_ALL}")
        if not files:
            print(f"{Fore.YELLOW}Warning: No files found matching the specified patterns{Style.RESET_ALL}")
            return 0
        
        # Sort files for consistent output
        files.sort()
        
        matches_found = False
        for file_index, file in enumerate(files):
            if verbose:
                progress = (file_index + 1) / len(files) * 100
                print(f"\r{Fore.CYAN}Searching file {file_index+1}/{len(files)} ({progress:.1f}%): {file}{Style.RESET_ALL}", end="")
                sys.stdout.flush()
                
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                file_matches = False
                for i, line in enumerate(lines, 1):
                    if regex.search(line):
                        if not file_matches:
                            print(f"{Fore.CYAN}{file}:{Style.RESET_ALL}")
                            file_matches = True
                            matches_found = True
                        
                        # Highlight matches in line
                        highlighted_line = line
                        if ignore_case:
                            # Case-insensitive highlighting is more complex
                            parts = []
                            last_end = 0
                            for match in regex.finditer(line):
                                start, end = match.span()
                                parts.append(line[last_end:start])
                                parts.append(f"{Fore.YELLOW}{line[start:end]}{Style.RESET_ALL}")
                                last_end = end
                            parts.append(line[last_end:])
                            highlighted_line = "".join(parts)
                        else:
                            # Simple case-sensitive highlighting
                            highlighted_line = regex.sub(f"{Fore.YELLOW}\\g<0>{Style.RESET_ALL}", line)
                        
                        if line_numbers:
                            print(f"{Fore.GREEN}{i:4d}{Style.RESET_ALL}: {highlighted_line}", end="")
                            if not highlighted_line.endswith('\n'):
                                print()
                        else:
                            print(f"  {highlighted_line}", end="")
                            if not highlighted_line.endswith('\n'):
                                print()
            except Exception as e:
                print(f"{Fore.RED}Error reading {file}: {str(e)}{Style.RESET_ALL}")
        
        if verbose:
            print(f"\n{Fore.CYAN}================================{Style.RESET_ALL}")
            
        if not matches_found:
            print(f"{Fore.YELLOW}No matches found in any of the specified files.{Style.RESET_ALL}")
        return 0 if matches_found else 1

    def apply_diff(self, args: argparse.Namespace) -> int:
        """
        Apply changes to a file using diff-like syntax.

        Args:
            args: Command-line arguments

        Returns:
            int: Exit code (0 for success, non-zero for errors)
        """
        file_path = args.file
        search_text = args.search
        replace_text = args.replace
        preview = args.preview
        verbose = getattr(args, 'verbose', False)
        
        if verbose:
            print(f"{Fore.CYAN}========== APPLY DIFF =========={Style.RESET_ALL}")
            print(f"{Fore.YELLOW}File:{Style.RESET_ALL} {file_path}")
            print(f"{Fore.YELLOW}Preview only:{Style.RESET_ALL} {preview}")
            print(f"{Fore.YELLOW}Search text length:{Style.RESET_ALL} {len(search_text)} characters")
            print(f"{Fore.YELLOW}Replace text length:{Style.RESET_ALL} {len(replace_text)} characters")
            print(f"{Fore.CYAN}=============================={Style.RESET_ALL}")
        # Replace escaped newlines with actual newlines
        search_text = search_text.replace('\\n', '\n')
        replace_text = replace_text.replace('\\n', '\n')
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"{Fore.RED}Error: File '{file_path}' does not exist{Style.RESET_ALL}")
            return 1
        
        try:
            # Read the file content
            if verbose:
                print(f"\n{Fore.YELLOW}Reading file content...{Style.RESET_ALL}")
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if verbose:
                print(f"{Fore.GREEN}Read {len(content)} characters from file{Style.RESET_ALL}")
            # Check if search text exists in the file
            if search_text not in content:
                print(f"{Fore.RED}Error: Search text not found in '{file_path}'{Style.RESET_ALL}")
                return 1
            
            # Create the modified content
            if verbose:
                print(f"\n{Fore.YELLOW}Applying changes...{Style.RESET_ALL}")
                
            modified_content = content.replace(search_text, replace_text)
            
            if verbose:
                instances = content.count(search_text)
                print(f"{Fore.GREEN}Replaced {instances} instance(s) of the search text{Style.RESET_ALL}")
            # Show diff preview
            if preview or True:  # Always show preview
                print(f"{Fore.CYAN}Diff preview for '{file_path}':{Style.RESET_ALL}")
                
                # Generate diff
                original_lines = content.splitlines()
                modified_lines = modified_content.splitlines()
                
                differ = difflib.unified_diff(
                    original_lines,
                    modified_lines,
                    fromfile=f'a/{file_path}',
                    tofile=f'b/{file_path}',
                    lineterm=''
                )
                
                # Display colored diff
                for line in differ:
                    if line.startswith('+'):
                        print(f"{Fore.GREEN}{line}{Style.RESET_ALL}")
                    elif line.startswith('-'):
                        print(f"{Fore.RED}{line}{Style.RESET_ALL}")
                    elif line.startswith('@'):
                        print(f"{Fore.CYAN}{line}{Style.RESET_ALL}")
                    else:
                        print(line)
            
            # Apply changes if not in preview mode
            if not preview:
                if verbose:
                    print(f"\n{Fore.YELLOW}Writing changes to file...{Style.RESET_ALL}")
                    
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                    
                print(f"{Fore.GREEN}Changes applied to '{file_path}'{Style.RESET_ALL}")
                
                if verbose:
                    print(f"{Fore.CYAN}=============================={Style.RESET_ALL}")
            return 0
        
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
            return 1

    def create_file(self, args: argparse.Namespace) -> int:
        """
        Create a new file with code content.

        Args:
            args: Command-line arguments

        Returns:
            int: Exit code (0 for success, non-zero for errors)
        """
        file_path = args.file
        content = args.content
        from_stdin = args.from_stdin
        force = args.force
        verbose = getattr(args, 'verbose', False)
        
        if verbose:
            print(f"{Fore.CYAN}========== CREATE FILE =========={Style.RESET_ALL}")
            print(f"{Fore.YELLOW}File:{Style.RESET_ALL} {file_path}")
            print(f"{Fore.YELLOW}From stdin:{Style.RESET_ALL} {from_stdin}")
            print(f"{Fore.YELLOW}Force overwrite:{Style.RESET_ALL} {force}")
            if content and not from_stdin:
                print(f"{Fore.YELLOW}Content length:{Style.RESET_ALL} {len(content)} characters")
            print(f"{Fore.CYAN}=============================={Style.RESET_ALL}")
        # Check if file exists and we're not forcing overwrite
        if os.path.exists(file_path) and not force:
            print(f"{Fore.RED}Error: File '{file_path}' already exists. Use --force to overwrite.{Style.RESET_ALL}")
            return 1
        
        # Get content from stdin if specified
        if from_stdin:
            print(f"Enter content for {file_path} (Ctrl+D to finish):")
            content = sys.stdin.read()
        elif content is None:
            content = ""
        
        try:
            # Create parent directories if they don't exist
            if verbose:
                print(f"\n{Fore.YELLOW}Creating parent directories if needed...{Style.RESET_ALL}")
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Write content to file
            if verbose:
                print(f"{Fore.YELLOW}Writing content to file...{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}File path:{Style.RESET_ALL} {os.path.abspath(file_path)}")
                print(f"{Fore.YELLOW}Content size:{Style.RESET_ALL} {len(content)} bytes")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            if verbose:
                print(f"{Fore.GREEN}File written successfully!{Style.RESET_ALL}")
                print(f"{Fore.CYAN}=============================={Style.RESET_ALL}")
            
            print(f"{Fore.GREEN}File '{file_path}' created successfully.{Style.RESET_ALL}")
            return 0
            
        except Exception as e:
            print(f"{Fore.RED}Error creating file '{file_path}': {str(e)}{Style.RESET_ALL}")
            return 1

    def display_file(self, args: argparse.Namespace) -> int:
        """
        Display file contents with optional line numbers.

        Args:
            args: Command-line arguments

        Returns:
            int: Exit code (0 for success, non-zero for errors)
        """
        file_path = args.file
        show_line_numbers = args.line_numbers
        line_range = args.range
        highlight_pattern = args.highlight
        verbose = getattr(args, 'verbose', False)
        
        if verbose:
            print(f"{Fore.CYAN}========== DISPLAY FILE =========={Style.RESET_ALL}")
            print(f"{Fore.YELLOW}File:{Style.RESET_ALL} {file_path}")
            print(f"{Fore.YELLOW}Show line numbers:{Style.RESET_ALL} {show_line_numbers}")
            print(f"{Fore.YELLOW}Line range:{Style.RESET_ALL} {line_range if line_range else 'All lines'}")
            print(f"{Fore.YELLOW}Highlight pattern:{Style.RESET_ALL} {highlight_pattern if highlight_pattern else 'None'}")
            print(f"{Fore.CYAN}================================{Style.RESET_ALL}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"{Fore.RED}Error: File '{file_path}' does not exist{Style.RESET_ALL}")
            return 1
        
        try:
            # Read the file content
            if verbose:
                print(f"\n{Fore.YELLOW}Reading file content...{Style.RESET_ALL}")
                
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            if verbose:
                file_size = os.path.getsize(file_path)
                line_count = len(lines)
                print(f"{Fore.GREEN}Read {file_size} bytes, {line_count} lines from file{Style.RESET_ALL}")
            
            # Parse line range if specified
            start_line, end_line = 1, len(lines)
            if line_range:
                try:
                    if '-' in line_range:
                        start_line, end_line = map(int, line_range.split('-'))
                    else:
                        start_line = end_line = int(line_range)
                    
                    # Adjust to 0-based indexing
                    start_line = max(1, start_line)
                    end_line = min(len(lines), end_line)
                    
                    if verbose:
                        print(f"{Fore.YELLOW}Displaying lines:{Style.RESET_ALL} {start_line} to {end_line} (out of {len(lines)})")
                except ValueError:
                    print(f"{Fore.RED}Error: Invalid line range '{line_range}'{Style.RESET_ALL}")
                    return 1
            
            # Compile highlight regex if specified
            highlight_regex = None
            if highlight_pattern:
                try:
                    if verbose:
                        print(f"{Fore.YELLOW}Compiling highlight pattern...{Style.RESET_ALL}")
                    highlight_regex = re.compile(highlight_pattern)
                    if verbose:
                        print(f"{Fore.GREEN}Pattern compiled successfully{Style.RESET_ALL}")
                except re.error as e:
                    print(f"{Fore.RED}Error in highlight pattern: {str(e)}{Style.RESET_ALL}")
                    return 1
            
            # Display file content with appropriate formatting
            if verbose:
                print(f"\n{Fore.YELLOW}Displaying file content...{Style.RESET_ALL}")
                print(f"{Fore.CYAN}================================{Style.RESET_ALL}")
                
            print(f"{Fore.CYAN}=== {file_path} ==={Style.RESET_ALL}")
            
            line_count = 0
            for i, line in enumerate(lines, 1):
                if start_line <= i <= end_line:
                    # Highlight text if pattern is specified
                    if highlight_regex and highlight_regex.search(line):
                        highlighted_line = highlight_regex.sub(
                            f"{Fore.YELLOW}\\g<0>{Style.RESET_ALL}", line
                        )
                    else:
                        highlighted_line = line
                    
                    # Display with or without line numbers
                    if show_line_numbers:
                        print(f"{Fore.GREEN}{i:4d}{Style.RESET_ALL}| {highlighted_line}", end="")
                        if not highlighted_line.endswith('\n'):
                            print()
                    else:
                        print(highlighted_line, end="")
                        if not highlighted_line.endswith('\n'):
                            print()
                    
                    line_count += 1
                    # Show progress for larger files
                    if verbose and len(lines) > 100 and line_count % 50 == 0:
                        progress = (i - start_line + 1) / (end_line - start_line + 1) * 100
                        print(f"{Fore.CYAN}Progress: {progress:.1f}% ({i}/{end_line}){Style.RESET_ALL}")
            
            if verbose:
                print(f"\n{Fore.GREEN}Displayed {line_count} lines from file{Style.RESET_ALL}")
                print(f"{Fore.CYAN}================================{Style.RESET_ALL}")
            
            return 0
            
        except Exception as e:
            print(f"{Fore.RED}Error displaying file '{file_path}': {str(e)}{Style.RESET_ALL}")
            return 1


if __name__ == "__main__":
    editor = CodeEditor()
    sys.exit(editor.run())
