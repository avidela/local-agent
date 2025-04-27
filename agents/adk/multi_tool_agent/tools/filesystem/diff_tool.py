# diff_tool.py
# This file contains a function to use the system's diff command via subprocess.

import subprocess
import os
from typing import Optional

def diff_files(file_path1: str, file_path2: str) -> dict:
    """
    Uses the system's diff command to compare two files and show differences.

    Assumes input file paths are relative to the directory where files are written
    by tools like `write_file` (which appears to be /repos).

    Args:
        file_path1: The path to the first file, relative to the /repos directory.
        file_path2: The path to the second file, relative to the /repos directory.

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

        # Adjust file paths to be relative to the subprocess cwd (/app)
        # by prepending ../repos/ assuming input paths are relative to /repos
        adjusted_file_path1 = f"../repos/{file_path1}"
        debug_info['adjusted_file_path1'] = adjusted_file_path1

        adjusted_file_path2 = f"../repos/{file_path2}"
        debug_info['adjusted_file_path2'] = adjusted_file_path2

        # Construct the diff command (using unified format -u)
        command = ["diff", "-u", adjusted_file_path1, adjusted_file_path2]
        debug_info['command_executed'] = " ".join(command) # For debugging

        # Execute the diff command, explicitly setting the current working directory to /app
        # This is where the agent code runs, and where the subprocess is launched from.
        result = subprocess.run(command, capture_output=True, text=True, check=True, cwd='.') # cwd='.' is /app

        # Handle diff exit codes:
        # 0: Files are identical
        # 1: Files differ
        # >1: Error
        debug_info['return_code'] = result.returncode
        debug_info['stderr'] = result.stderr.strip()
        debug_info['stdout'] = result.stdout.strip() # Include stdout in debug for non-zero return codes


        if result.returncode == 0:
            return {'status': 'success', 'output': "Files are identical.", 'debug': debug_info}
        elif result.returncode == 1:
            return {'status': 'success', 'output': result.stdout.strip(), 'debug': debug_info}
        else:
            # Error occurred
            return {'status': 'error', 'message': f"Error executing diff command: {result.stderr.strip()}", 'debug': debug_info}

    except FileNotFoundError:
        # Handle case where the diff command itself is not found
        debug_info['error_type'] = 'FileNotFoundError'
        return {'status': 'error', 'message': f"Error: diff command not found. Is diff installed and in your PATH?", 'debug': debug_info}
    except Exception as e:
        # Catch any other exceptions during the process
        debug_info['error_type'] = type(e).__name__
        debug_info['exception_message'] = str(e)
        return {'status': 'error', 'message': f"An internal error occurred while trying to diff files: {e}", 'debug': debug_info}
