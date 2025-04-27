# Agent Tools

The `local_agent` is equipped with a variety of tools to interact with the filesystem and perform other tasks. These tools are integrated using wrappers provided by the Agent Development Kit (ADK).

## Tool Wrappers

*   **`LangchainTool`**: Used to wrap standard tools provided by the Langchain community library.
*   **`FunctionTool`**: Used to wrap custom Python functions, making them callable by the agent.

All filesystem tools operate relative to the directory specified by the `REPO_ROOT` environment variable (see [Configuration](configuration.md)).

## Available Tools

### Standard Filesystem Tools (via `LangchainTool`)

These tools are wrappers around common file operations provided by `langchain_community.tools.file_management`.

*   **`FileSearchTool`**: Searches for files within the `REPO_ROOT`. *(Note: Specific arguments and behavior depend on the Langchain implementation.)*
*   **`ListDirectoryTool`**: Lists the contents of a directory within `REPO_ROOT`.
*   **`ReadFileTool`**: Reads the content of a file within `REPO_ROOT`.
*   **`WriteFileTool`**: Writes content to a file within `REPO_ROOT`. Can create or overwrite files.
*   **`CopyFileTool`**: Copies a file from one path to another within `REPO_ROOT`.
*   **`MoveFileTool`**: Moves or renames a file within `REPO_ROOT`.
*   **`DeleteFileTool`**: Deletes a file within `REPO_ROOT`.

*(For exact arguments and return values of these Langchain tools, refer to the Langchain documentation or inspect their usage within the agent's interactions.)*

### Custom Filesystem Tools (via `FunctionTool`)

These tools are custom Python functions wrapping system commands for more advanced filesystem interactions. They all return a dictionary with the following structure:

*   `status` (str): 'success' or 'error'.
*   `output` (str | list | None): The result of the command if successful (format depends on the tool). Present only if `status` is 'success'.
*   `message` (str | None): An error message if the command failed. Present only if `status` is 'error'.
*   `debug` (dict): Detailed information about the command execution (command string, paths, return code, stderr, stdout).

---

#### `grep_file`

*   **Wraps:** System `grep` command.
*   **Purpose:** Searches for a `pattern` within a specified `file_path` (or recursively in a directory).
*   **Arguments:**
    *   `file_path` (str): Path to file or directory relative to `REPO_ROOT`.
    *   `pattern` (str): The pattern to search for.
    *   `ignore_case` (bool, optional): Corresponds to `grep -i`.
    *   `whole_word` (bool, optional): Corresponds to `grep -w`.
    *   `show_line_numbers` (bool, optional): Corresponds to `grep -n`.
    *   `recursive` (bool, optional): Corresponds to `grep -r`.
    *   `invert_match` (bool, optional): Corresponds to `grep -v`.
    *   `count_matches` (bool, optional): Corresponds to `grep -c`.
    *   `files_with_matches` (bool, optional): Corresponds to `grep -l`.
*   **Returns (`output` on success):**
    *   A string containing the matching lines (with optional line numbers), counts per file, or filenames, depending on the flags used.
    *   A specific message string if no lines matched the pattern (e.g., "No lines found matching...").

---

#### `find_files`

*   **Wraps:** System `find` command.
*   **Purpose:** Finds files or directories matching criteria within a `dir_path`.
*   **Behavior:** **Excludes hidden files and directories** (starting with '.') by default using `-path '*/.*' -prune -o`.
*   **Arguments:**
    *   `dir_path` (str): Directory path relative to `REPO_ROOT` to start the search.
    *   `name_pattern` (str, optional): Filename pattern (e.g., `'*.py'`, `'README.md'`). Corresponds to `find -name`.
    *   `type` (str, optional): Type of entry ('f' for file, 'd' for directory). Corresponds to `find -type`.
*   **Returns (`output` on success):**
    *   A list of strings, where each string is a found path relative to the `dir_path` used in the search. Returns an empty list if no matches are found.

---

#### `diff_files`

*   **Wraps:** System `diff -u` command (unified format).
*   **Purpose:** Compares two files (`file_path1`, `file_path2`) and shows differences.
*   **Arguments:**
    *   `file_path1` (str): Path to the first file relative to `REPO_ROOT`.
    *   `file_path2` (str): Path to the second file relative to `REPO_ROOT`.
*   **Returns (`output` on success):**
    *   A string containing the unified diff output if the files differ.
    *   The exact string "Files are identical." if the files are identical.

