# find_tool.py
# This file contains a function to use the system's find command via subprocess.

import subprocess
import os
from typing import Optional # Import Optional

def find_files(dir_path: str, name_pattern: Optional[str] = None, type: Optional[str] = None) -> dict:
    """
    Uses the system's find command to search for files and directories,
    excluding hidden files/directories (starting with '.') by default.

    Args:
        dir_path: The directory to start the search from, relative to the repository root (/repos).
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

        # Get the repository root from the environment variable, defaulting to /repos
        repo_root = os.getenv("REPO_ROOT", "/repos")
        debug_info['repo_root'] = repo_root

        # Construct the full path to search by joining repo_root and dir_path
        full_search_path = os.path.join(repo_root, dir_path)
        debug_info['full_search_path'] = full_search_path

        # Construct the find command
        command = [
            "find",
            full_search_path
        ]

        # Add pruning for hidden files/directories BEFORE other filters
        # Match any path component starting with '.' and prune it.
        command.extend(["-path", "*/.*", "-prune", "-o"])

        # Add type filter if specified (applied only to non-pruned items)
        if type:
            if type.lower() in ['f', 'd']:
                command.extend(["-type", type.lower()])
            else:
                return {'status': 'error', 'message': f"Invalid type specified: {type}. Use 'f' for file or 'd' for directory.", 'debug': debug_info}

        # Add name pattern filter if specified (applied only to non-pruned items)
        if name_pattern:
             # Apply name pattern only to non-excluded items
            command.extend(["-name", name_pattern])

        # Add the print action at the end (applied only to non-pruned items matching filters)
        command.append("-print")

        debug_info['command_executed'] = " ".join(command) # For debugging

        # Execute the find command from the root directory
        result = subprocess.run(command, capture_output=True, text=True, check=True, cwd='/')

        found_paths = result.stdout.strip().splitlines()

        return {'status': 'success', 'output': found_paths, 'debug': debug_info}

    except subprocess.CalledProcessError as e:
        debug_info['return_code'] = e.returncode
        debug_info['stderr'] = e.stderr.strip()
        return {'status': 'error', 'message': f"Error executing find command for {dir_path}: {e.stderr.strip()}", 'debug': debug_info}
    except FileNotFoundError:
        debug_info['error_type'] = 'FileNotFoundError'
        return {'status': 'error', 'message': f"Error: find command not found. Is find installed and in your PATH?", 'debug': debug_info}
    except Exception as e:
        debug_info['error_type'] = type(e).__name__
        debug_info['exception_message'] = str(e)
        return {'status': 'error', 'message': f"An internal error occurred while trying to find files in {dir_path}: {e}", 'debug': debug_info}
