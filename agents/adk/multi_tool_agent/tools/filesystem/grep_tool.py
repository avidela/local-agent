# grep_tool.py
# This file contains a function to use the system's grep command via subprocess.

import subprocess
import os # Import the os module
from typing import Optional # Import Optional for type hints if needed later

def grep_file(file_path: str, pattern: str, ignore_case: bool = False, whole_word: bool = False, show_line_numbers: bool = False, recursive: bool = False, invert_match: bool = False, count_matches: bool = False, files_with_matches: bool = False) -> dict:
    """
    Uses the system's grep command to search for a pattern in a file(s) with optional flags.
    Interprets file_path relative to REPO_ROOT (defaulting to /repos).

    Args:
        file_path: The path to the file or directory to search in, relative to the repository root.
        pattern: The string pattern to search for.
        ignore_case: If True, perform case-insensitive matching (-i flag).
        whole_word: If True, match only whole words (-w flag).
        show_line_numbers: If True, print line number with output lines (-n flag).
        recursive: If True, search recursively in directories (-r flag).
        invert_match: If True, select non-matching lines (-v flag).
        count_matches: If True, print only a count of matching lines per file (-c flag).
        files_with_matches: If True, print only names of files containing matches (-l flag).

    Returns:
        A dictionary indicating the status and either the matching lines/counts/filenames
        or an error message, including debug information on error.
    """
    debug_info = {}
    try:
        # Get debug info before running the command
        debug_info['subprocess_cwd'] = os.getcwd()
        try:
            debug_info['subprocess_listdir'] = os.listdir('.')
        except Exception as listdir_e:
            debug_info['subprocess_listdir_error'] = str(listdir_e)

        # Get the repository root from the environment variable, defaulting to /repos
        repo_root = os.getenv("REPO_ROOT", "/repos")
        debug_info['repo_root'] = repo_root

        # Construct the full path to search by joining repo_root and file_path
        full_path = os.path.join(repo_root, file_path)
        debug_info['full_path_searched'] = full_path

        # Construct the grep command with all optional flags
        command = ["grep"]
        if ignore_case:
            command.append("-i")
        if whole_word:
            command.append("-w")
        if show_line_numbers:
            command.append("-n")
        if recursive:
            command.append("-r")
        if invert_match:
            command.append("-v")
        if count_matches:
            command.append("-c")
        if files_with_matches:
            command.append("-l")

        # Add the pattern and the full path
        command.extend([pattern, full_path])
        debug_info['command_executed'] = " ".join(command) # For debugging

        # Execute the grep command from the root directory
        # Using cwd='/' ensures paths are interpreted correctly
        result = subprocess.run(command, capture_output=True, text=True, check=False, cwd='/') # Use check=False to handle grep's exit codes manually

        debug_info['return_code'] = result.returncode
        debug_info['stderr'] = result.stderr.strip()
        debug_info['stdout'] = result.stdout.strip() # Capture stdout even if returncode is non-zero

        # Handle grep's exit codes:
        # 0: One or more lines were selected.
        # 1: No lines were selected.
        # >1: An error occurred.

        if result.returncode == 0:
            # Success, matches found (or all lines matched with -v)
            return {'status': 'success', 'output': result.stdout.strip(), 'debug': debug_info}
        elif result.returncode == 1:
            # No lines selected - this is not an error, just means no matches.
            if invert_match:
                 # -v was used, and no lines were selected (meaning all lines matched the pattern)
                 return {'status': 'success', 'output': f"All lines matched the pattern '{pattern}' in {file_path}.", 'debug': debug_info}
            else:
                 # No -v, and no lines were selected (meaning no lines matched the pattern)
                 return {'status': 'success', 'output': f"No lines found matching the pattern '{pattern}' in {file_path}.", 'debug': debug_info}
        else:
            # An actual error occurred (e.g., file not found, permissions)
            error_message = f"Error executing grep command for {file_path}: {result.stderr.strip()}"
            # Add specific message for file not found if stderr indicates it
            if "No such file or directory" in result.stderr:
                 error_message = f"Error: File or directory not found at specified path: {file_path} (resolved to {full_path})"
            return {'status': 'error', 'message': error_message, 'debug': debug_info}

    except FileNotFoundError:
        # Handle case where the grep command itself is not found
        debug_info['error_type'] = 'FileNotFoundError'
        return {'status': 'error', 'message': f"Error: grep command not found. Is grep installed and in your PATH?", 'debug': debug_info}
    except Exception as e:
        # Catch any other exceptions during the process
        debug_info['error_type'] = type(e).__name__
        debug_info['exception_message'] = str(e)
        return {'status': 'error', 'message': f"An internal error occurred while trying to grep {file_path}: {e}", 'debug': debug_info}

# Note: Integration into the agent's main logic is assumed to be handled elsewhere.
