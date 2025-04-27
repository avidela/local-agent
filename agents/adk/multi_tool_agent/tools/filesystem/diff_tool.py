# diff_tool.py
# This file contains a function to use the system's diff command via subprocess.

import subprocess
import os
from typing import Optional

def diff_files(file_path1: str, file_path2: str) -> dict:
    """
    Uses the system's diff command to compare two files and show differences.
    Interprets file paths relative to REPO_ROOT (defaulting to /repos).

    Args:
        file_path1: The path to the first file, relative to the repository root.
        file_path2: The path to the second file, relative to the repository root.

    Returns:
        A dictionary indicating the status and either the diff output (unified format)
        or a message indicating files are identical, or an error message.
        Includes debug information on error.
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

        # Construct the full paths by joining repo_root and the relative file paths
        full_path1 = os.path.join(repo_root, file_path1)
        debug_info['full_path1'] = full_path1

        full_path2 = os.path.join(repo_root, file_path2)
        debug_info['full_path2'] = full_path2

        # Construct the diff command (using unified format -u)
        command = ["diff", "-u", full_path1, full_path2]
        debug_info['command_executed'] = " ".join(command) # For debugging

        # Execute the diff command from the root directory
        # Using cwd='/' ensures paths are interpreted correctly
        # Use check=False to handle diff's exit codes manually
        result = subprocess.run(command, capture_output=True, text=True, check=False, cwd='/')

        debug_info['return_code'] = result.returncode
        debug_info['stderr'] = result.stderr.strip()
        debug_info['stdout'] = result.stdout.strip() # Capture stdout even if returncode is non-zero

        # Handle diff exit codes:
        # 0: Files are identical.
        # 1: Files differ.
        # >1: Error.

        if result.returncode == 0:
            # Files are identical
            return {'status': 'success', 'output': "Files are identical.", 'debug': debug_info}
        elif result.returncode == 1:
            # Files differ - this is a successful outcome showing differences
            return {'status': 'success', 'output': result.stdout.strip(), 'debug': debug_info}
        else:
            # An actual error occurred (e.g., file not found, permissions)
            error_message = f"Error executing diff command: {result.stderr.strip()}"
            # Add specific message for file not found if stderr indicates it
            if "No such file or directory" in result.stderr:
                 # Check which file might be missing
                 missing_file = ""
                 if not os.path.exists(full_path1): # Check existence from the perspective of the agent's environment
                     missing_file = file_path1
                 elif not os.path.exists(full_path2):
                     missing_file = file_path2

                 if missing_file:
                     error_message = f"Error: File not found at specified path: {missing_file}"
                 else: # If both seem to exist from agent's view, report the stderr
                     error_message = f"Error executing diff command (file possibly not found at {full_path1} or {full_path2}): {result.stderr.strip()}"

            return {'status': 'error', 'message': error_message, 'debug': debug_info}

    except FileNotFoundError:
        # Handle case where the diff command itself is not found
        debug_info['error_type'] = 'FileNotFoundError'
        return {'status': 'error', 'message': f"Error: diff command not found. Is diff installed and in your PATH?", 'debug': debug_info}
    except Exception as e:
        # Catch any other exceptions during the process
        debug_info['error_type'] = type(e).__name__
        debug_info['exception_message'] = str(e)
        return {'status': 'error', 'message': f"An internal error occurred while trying to diff files: {e}", 'debug': debug_info}

# Note: Integration into the agent's main logic is assumed to be handled elsewhere.
