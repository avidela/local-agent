# find_tool.py
# This file contains a function to use the system's find command via subprocess.

import subprocess
import os
from typing import Optional # Import Optional

def find_files(dir_path: str, name_pattern: Optional[str] = None, type: Optional[str] = None) -> dict:
    """
    Uses the system's find command to search for files and directories.

    Args:
        dir_path: The directory to start the search from.
        name_pattern: A pattern to match filenames (e.g., '*.py').
        type: The type of entry to find ('f' for file, 'd' for directory).

    Returns:
        A dictionary indicating the status and either the list of found paths
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

        # Adjust dir_path for the subprocess environment if it starts with 'local-agent/'
        adjusted_dir_path = dir_path
        if adjusted_dir_path.startswith('local-agent/'):
            adjusted_dir_path = adjusted_dir_path[len('local-agent/'):]
        debug_info['adjusted_dir_path'] = adjusted_dir_path

        # Construct the find command
        command = ["find", adjusted_dir_path]

        if name_pattern:
            command.extend(["-name", name_pattern])
        if type:
            if type.lower() in ['f', 'd']:
                command.extend(["-type", type.lower()])
            else:
                return {'status': 'error', 'message': f"Invalid type specified: {type}. Use 'f' for file or 'd' for directory.", 'debug': debug_info}

        debug_info['command_executed'] = " ".join(command) # For debugging

        # Execute the find command
        # check=True will raise CalledProcessError for non-zero exit codes
        result = subprocess.run(command, capture_output=True, text=True, check=True, cwd='.')

        # If check=True, this part is only reached if returncode is 0 (matches found or no matches)
        # find returns 0 even if no matches are found, output is just empty.
        found_paths = result.stdout.strip().splitlines()
        return {'status': 'success', 'output': found_paths, 'debug': debug_info}

    except subprocess.CalledProcessError as e:
        # This handles cases where find exits with a non-zero status (usually an error)
        debug_info['return_code'] = e.returncode
        debug_info['stderr'] = e.stderr.strip()
        return {'status': 'error', 'message': f"Error executing find command for {dir_path}: {e.stderr.strip()}", 'debug': debug_info}
    except FileNotFoundError:
        # Handle case where the find command itself is not found
        debug_info['error_type'] = 'FileNotFoundError'
        return {'status': 'error', 'message': f"Error: find command not found. Is find installed and in your PATH?", 'debug': debug_info}
    except Exception as e:
        # Catch any other exceptions during the process
        debug_info['error_type'] = type(e).__name__
        debug_info['exception_message'] = str(e)
        return {'status': 'error', 'message': f"An internal error occurred while trying to find files in {dir_path}: {e}", 'debug': debug_info}
