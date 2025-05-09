# grep_tool.py
# This file contains a function to use the system's grep command via subprocess.

import subprocess
import os # Import the os module for debugging info

def grep_file(file_path: str, pattern: str, ignore_case: bool = False, whole_word: bool = False, show_line_numbers: bool = False, recursive: bool = False, invert_match: bool = False, count_matches: bool = False, files_with_matches: bool = False) -> dict:
    """
    Uses the system's grep command to search for a pattern in a file(s) with optional flags.

    Args:
        file_path: The path to the file or directory to search in.
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

        # Adjust file_path for the subprocess environment if it starts with 'local-agent/'
        # This assumes the subprocess cwd is the same as the local-agent root.
        adjusted_file_path = file_path
        if adjusted_file_path.startswith('local-agent/'):
            adjusted_file_path = adjusted_file_path[len('local-agent/'):]
        debug_info['adjusted_file_path'] = adjusted_file_path

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

        # Add the pattern and file path
        command.extend([pattern, adjusted_file_path])
        debug_info['command_executed'] = " ".join(command) # For debugging

        # Execute the grep command, explicitly setting the current working directory
        # We will keep cwd='.' as it seems to be the /app directory
        result = subprocess.run(command, capture_output=True, text=True, check=True, cwd='.')

        # If check=True, this part is only reached if returncode is 0 (matches found or no matches with -v)
        return {'status': 'success', 'output': result.stdout.strip(), 'debug': debug_info}

    except subprocess.CalledProcessError as e:
        # This handles cases where grep exits with a non-zero status.
        # returncode 1 means no matches were found (unless -v is used).
        # returncode > 1 means an actual error occurred (e.g., file not found).
        debug_info['return_code'] = e.returncode
        debug_info['stderr'] = e.stderr.strip()

        # Grep returns 1 for no matches. If -v is used, return 0 for no matches.
        # We should consider returncode 1 as a successful outcome if no matches were expected or found.
        if e.returncode == 1 and not invert_match:
             # No matches found and -v was not used
             return {'status': 'success', 'output': f"No lines found matching the pattern '{pattern}' in {file_path}.", 'debug': debug_info}
        elif e.returncode == 0 and invert_match:
             # No non-matching lines found when -v was used
             return {'status': 'success', 'output': f"All lines matched the pattern '{pattern}' in {file_path}.", 'debug': debug_info}
        elif e.returncode == 1 and invert_match:
             # Some non-matching lines found when -v was used (grep returns 1 if no lines selected)
             # This case might need refinement based on exact grep -v behavior and desired output
             # For now, let's treat it as success with output if stderr is empty, otherwise error
             if e.stderr.strip() == "":
                  return {'status': 'success', 'output': result.stdout.strip(), 'debug': debug_info}
             else:
                  return {'status': 'error', 'message': f"Error executing grep command for {file_path}: {e.stderr.strip()}", 'debug': debug_info}
        else:
            # Other non-zero exit codes indicate an error
            return {'status': 'error', 'message': f"Error executing grep command for {file_path}: {e.stderr.strip()}", 'debug': debug_info}
    except FileNotFoundError:
        # Handle case where the grep command itself is not found
        debug_info['error_type'] = 'FileNotFoundError'
        return {'status': 'error', 'message': f"Error: grep command not found. Is grep installed and in your PATH?", 'debug': debug_info}
    except Exception as e:
        # Catch any other exceptions during the process
        debug_info['error_type'] = type(e).__name__
        debug_info['exception_message'] = str(e)
        return {'status': 'error', 'message': f"An internal error occurred while trying to grep {file_path}: {e}", 'debug': debug_info}

# Note: This function needs to be integrated into the agent's main logic
# to be callable based on user requests. This involves modifying agent.py
# or a similar file to recognize grep requests and call this function.
# We also need to update the prompt to tell the agent about this new capability.
