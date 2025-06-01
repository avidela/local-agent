"""
Search tools for PydanticAI agents.
"""

import subprocess
from pathlib import Path
from typing import List, Optional
from ..base import BaseTool, ToolMetadata, ToolResult


class FileSearchTool(BaseTool):
    """Tool for searching files by pattern."""
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="file_search",
            description="Search for files matching a pattern",
            parameters={
                "pattern": {
                    "type": "str",
                    "description": "File name pattern to search for (e.g., '*.py', 'README.md')",
                    "required": True
                },
                "dir_path": {
                    "type": "str",
                    "description": "Directory path to search in",
                    "required": False,
                    "default": "."
                }
            },
            examples=["file_search(pattern='*.py', dir_path='src/')"],
            category="filesystem"
        )
    
    async def execute(self, pattern: str, dir_path: str = ".") -> ToolResult:
        """Search for files matching pattern."""
        try:
            base_path = Path(self.config.get("base_path", "."))
            search_path = base_path / dir_path
            
            # Security check
            if not str(search_path.resolve()).startswith(str(base_path.resolve())):
                return ToolResult(
                    status="error",
                    output=None,
                    message="Access denied: Search path outside base directory",
                    debug={"dir_path": dir_path, "base_path": str(base_path)}
                )
            
            matches = []
            for file_path in search_path.rglob(pattern):
                if file_path.is_file():
                    relative_path = file_path.relative_to(base_path)
                    matches.append(str(relative_path))
            
            return ToolResult(
                status="success",
                output=matches,
                message=None,
                debug={
                    "pattern": pattern,
                    "dir_path": dir_path,
                    "matches_found": len(matches)
                }
            )
            
        except Exception as e:
            return ToolResult(
                status="error",
                output=None,
                message=f"File search failed: {str(e)}",
                debug={
                    "pattern": pattern,
                    "dir_path": dir_path,
                    "error_type": type(e).__name__
                }
            )


class GrepFileTool(BaseTool):
    """Tool for searching text within files using grep."""
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="grep_file",
            description="Search for text patterns within files",
            parameters={
                "pattern": {
                    "type": "str",
                    "description": "Text pattern to search for",
                    "required": True
                },
                "file_path": {
                    "type": "str",
                    "description": "File or directory path to search in",
                    "required": True
                },
                "ignore_case": {
                    "type": "bool",
                    "description": "Ignore case when matching",
                    "required": False,
                    "default": False
                },
                "show_line_numbers": {
                    "type": "bool",
                    "description": "Show line numbers in results",
                    "required": False,
                    "default": True
                },
                "recursive": {
                    "type": "bool",
                    "description": "Search recursively in directories",
                    "required": False,
                    "default": False
                },
                "whole_word": {
                    "type": "bool",
                    "description": "Match whole words only",
                    "required": False,
                    "default": False
                }
            },
            examples=["grep_file(pattern='TODO', file_path='src/', recursive=True)"],
            category="filesystem"
        )
    
    async def execute(
        self,
        pattern: str,
        file_path: str,
        ignore_case: bool = False,
        show_line_numbers: bool = True,
        recursive: bool = False,
        whole_word: bool = False
    ) -> ToolResult:
        """Search for text patterns in files."""
        try:
            base_path = Path(self.config.get("base_path", "."))
            target_path = base_path / file_path
            
            # Security check
            if not str(target_path.resolve()).startswith(str(base_path.resolve())):
                return ToolResult(
                    status="error",
                    output=None,
                    message="Access denied: Target path outside base directory",
                    debug={"file_path": file_path, "base_path": str(base_path)}
                )
            
            if not target_path.exists():
                return ToolResult(
                    status="error",
                    output=None,
                    message=f"Path not found: {file_path}",
                    debug={"file_path": file_path}
                )
            
            # Build grep command
            cmd = ["grep"]
            if ignore_case:
                cmd.append("-i")
            if show_line_numbers:
                cmd.append("-n")
            if recursive:
                cmd.append("-r")
            if whole_word:
                cmd.append("-w")
            
            cmd.extend([pattern, str(target_path)])
            
            # Execute grep command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=base_path
            )
            
            if result.returncode == 0:
                # Found matches
                output = result.stdout.strip()
                lines = output.split('\n') if output else []
                
                return ToolResult(
                    status="success",
                    output=lines,
                    message=None,
                    debug={
                        "pattern": pattern,
                        "file_path": file_path,
                        "matches_found": len(lines),
                        "command": " ".join(cmd)
                    }
                )
            elif result.returncode == 1:
                # No matches found
                return ToolResult(
                    status="success",
                    output=[],
                    message=f"No lines found matching pattern '{pattern}'",
                    debug={
                        "pattern": pattern,
                        "file_path": file_path,
                        "matches_found": 0,
                        "command": " ".join(cmd)
                    }
                )
            else:
                # Error occurred
                return ToolResult(
                    status="error",
                    output=None,
                    message=f"Grep command failed: {result.stderr.strip()}",
                    debug={
                        "pattern": pattern,
                        "file_path": file_path,
                        "command": " ".join(cmd),
                        "return_code": result.returncode,
                        "stderr": result.stderr.strip()
                    }
                )
                
        except Exception as e:
            return ToolResult(
                status="error",
                output=None,
                message=f"Grep operation failed: {str(e)}",
                debug={
                    "pattern": pattern,
                    "file_path": file_path,
                    "error_type": type(e).__name__
                }
            )


class FindFilesTool(BaseTool):
    """Tool for finding files using the find command."""
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="find_files",
            description="Find files or directories matching criteria",
            parameters={
                "dir_path": {
                    "type": "str",
                    "description": "Directory path to search in",
                    "required": True
                },
                "name_pattern": {
                    "type": "str",
                    "description": "Name pattern to match (e.g., '*.py', 'README.md')",
                    "required": False
                },
                "file_type": {
                    "type": "str",
                    "description": "Type of entry to find ('f' for file, 'd' for directory)",
                    "required": False
                }
            },
            examples=["find_files(dir_path='src/', name_pattern='*.py', file_type='f')"],
            category="filesystem"
        )
    
    async def execute(
        self,
        dir_path: str,
        name_pattern: Optional[str] = None,
        file_type: Optional[str] = None
    ) -> ToolResult:
        """Find files or directories matching criteria."""
        try:
            base_path = Path(self.config.get("base_path", "."))
            search_path = base_path / dir_path
            
            # Security check
            if not str(search_path.resolve()).startswith(str(base_path.resolve())):
                return ToolResult(
                    status="error",
                    output=None,
                    message="Access denied: Search path outside base directory",
                    debug={"dir_path": dir_path, "base_path": str(base_path)}
                )
            
            if not search_path.exists():
                return ToolResult(
                    status="error",
                    output=None,
                    message=f"Directory not found: {dir_path}",
                    debug={"dir_path": dir_path}
                )
            
            # Build find command
            cmd = ["find", str(search_path)]
            
            # Exclude hidden files/directories by default
            cmd.extend(["-path", "*/.*", "-prune", "-o"])
            
            # Add type filter
            if file_type:
                cmd.extend(["-type", file_type])
            
            # Add name pattern
            if name_pattern:
                cmd.extend(["-name", name_pattern])
            
            # Print results
            cmd.append("-print")
            
            # Execute find command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=base_path
            )
            
            if result.returncode == 0:
                output = result.stdout.strip()
                if output:
                    lines = output.split('\n')
                    # Convert absolute paths to relative paths
                    relative_paths = []
                    for line in lines:
                        try:
                            abs_path = Path(line)
                            if abs_path.exists():
                                rel_path = abs_path.relative_to(search_path)
                                relative_paths.append(str(rel_path))
                        except ValueError:
                            # Skip if can't make relative
                            continue
                    
                    return ToolResult(
                        status="success",
                        output=relative_paths,
                        message=None,
                        debug={
                            "dir_path": dir_path,
                            "name_pattern": name_pattern,
                            "file_type": file_type,
                            "matches_found": len(relative_paths),
                            "command": " ".join(cmd)
                        }
                    )
                else:
                    return ToolResult(
                        status="success",
                        output=[],
                        message="No files found matching criteria",
                        debug={
                            "dir_path": dir_path,
                            "name_pattern": name_pattern,
                            "file_type": file_type,
                            "matches_found": 0,
                            "command": " ".join(cmd)
                        }
                    )
            else:
                return ToolResult(
                    status="error",
                    output=None,
                    message=f"Find command failed: {result.stderr.strip()}",
                    debug={
                        "dir_path": dir_path,
                        "name_pattern": name_pattern,
                        "file_type": file_type,
                        "command": " ".join(cmd),
                        "return_code": result.returncode,
                        "stderr": result.stderr.strip()
                    }
                )
                
        except Exception as e:
            return ToolResult(
                status="error",
                output=None,
                message=f"Find operation failed: {str(e)}",
                debug={
                    "dir_path": dir_path,
                    "name_pattern": name_pattern,
                    "file_type": file_type,
                    "error_type": type(e).__name__
                }
            )