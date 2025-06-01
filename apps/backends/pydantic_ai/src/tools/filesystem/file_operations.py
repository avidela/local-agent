"""
File operation tools for PydanticAI agents.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
from ..base import BaseTool, ToolMetadata, ToolResult


class ReadFileTool(BaseTool):
    """Tool for reading file contents."""
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="read_file",
            description="Read the contents of a file",
            parameters={
                "file_path": {
                    "type": "str",
                    "description": "Path to the file to read",
                    "required": True
                }
            },
            examples=["read_file(file_path='README.md')"],
            category="filesystem"
        )
    
    async def execute(self, file_path: str) -> ToolResult:
        """Read file contents."""
        try:
            # Get base path from config or use current working directory
            base_path = Path(self.config.get("base_path", os.getcwd()))
            full_path = base_path / file_path
            
            # Security check - ensure path is within base_path
            if not str(full_path.resolve()).startswith(str(base_path.resolve())):
                return ToolResult(
                    status="error",
                    output=None,
                    message=f"Access denied: Path outside base directory",
                    debug={"file_path": file_path, "base_path": str(base_path)}
                )
            
            if not full_path.exists():
                return ToolResult(
                    status="error",
                    output=None,
                    message=f"File not found: {file_path}",
                    debug={"file_path": file_path, "full_path": str(full_path)}
                )
            
            if not full_path.is_file():
                return ToolResult(
                    status="error",
                    output=None,
                    message=f"Path is not a file: {file_path}",
                    debug={"file_path": file_path, "full_path": str(full_path)}
                )
            
            content = full_path.read_text(encoding='utf-8')
            return ToolResult(
                status="success",
                output=content,
                message=None,
                debug={"file_path": file_path, "size": len(content)}
            )
            
        except UnicodeDecodeError:
            return ToolResult(
                status="error", 
                output=None,
                message=f"Cannot read file - not a text file: {file_path}",
                debug={"file_path": file_path}
            )
        except Exception as e:
            return ToolResult(
                status="error",
                output=None,
                message=f"Failed to read file: {str(e)}",
                debug={"file_path": file_path, "error_type": type(e).__name__}
            )


class WriteFileTool(BaseTool):
    """Tool for writing content to files."""
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="write_file",
            description="Write content to a file",
            parameters={
                "file_path": {
                    "type": "str", 
                    "description": "Path to the file to write",
                    "required": True
                },
                "content": {
                    "type": "str",
                    "description": "Content to write to the file",
                    "required": True
                },
                "create_dirs": {
                    "type": "bool",
                    "description": "Create parent directories if they don't exist",
                    "required": False,
                    "default": True
                }
            },
            examples=["write_file(file_path='output.txt', content='Hello World')"],
            category="filesystem"
        )
    
    async def execute(self, file_path: str, content: str, create_dirs: bool = True) -> ToolResult:
        """Write content to file."""
        try:
            base_path = Path(self.config.get("base_path", os.getcwd()))
            full_path = base_path / file_path
            
            # Security check
            if not str(full_path.resolve()).startswith(str(base_path.resolve())):
                return ToolResult(
                    status="error",
                    output=None,
                    message="Access denied: Path outside base directory",
                    debug={"file_path": file_path, "base_path": str(base_path)}
                )
            
            # Create parent directories if needed
            if create_dirs:
                full_path.parent.mkdir(parents=True, exist_ok=True)
            
            full_path.write_text(content, encoding='utf-8')
            
            return ToolResult(
                status="success",
                output=f"Successfully wrote {len(content)} characters to {file_path}",
                message=None,
                debug={
                    "file_path": file_path,
                    "size": len(content),
                    "created_dirs": create_dirs
                }
            )
            
        except Exception as e:
            return ToolResult(
                status="error",
                output=None,
                message=f"Failed to write file: {str(e)}",
                debug={"file_path": file_path, "error_type": type(e).__name__}
            )


class ListDirectoryTool(BaseTool):
    """Tool for listing directory contents."""
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="list_directory",
            description="List the contents of a directory",
            parameters={
                "dir_path": {
                    "type": "str",
                    "description": "Path to the directory to list",
                    "required": True
                },
                "show_hidden": {
                    "type": "bool",
                    "description": "Include hidden files and directories",
                    "required": False,
                    "default": False
                }
            },
            examples=["list_directory(dir_path='src/')"],
            category="filesystem"
        )
    
    async def execute(self, dir_path: str, show_hidden: bool = False) -> ToolResult:
        """List directory contents."""
        try:
            base_path = Path(self.config.get("base_path", os.getcwd()))
            full_path = base_path / dir_path
            
            # Security check
            if not str(full_path.resolve()).startswith(str(base_path.resolve())):
                return ToolResult(
                    status="error",
                    output=None,
                    message="Access denied: Path outside base directory",
                    debug={"dir_path": dir_path, "base_path": str(base_path)}
                )
            
            if not full_path.exists():
                return ToolResult(
                    status="error",
                    output=None,
                    message=f"Directory not found: {dir_path}",
                    debug={"dir_path": dir_path, "full_path": str(full_path)}
                )
            
            if not full_path.is_dir():
                return ToolResult(
                    status="error",
                    output=None,
                    message=f"Path is not a directory: {dir_path}",
                    debug={"dir_path": dir_path, "full_path": str(full_path)}
                )
            
            contents = []
            for item in full_path.iterdir():
                if not show_hidden and item.name.startswith('.'):
                    continue
                    
                item_type = "directory" if item.is_dir() else "file"
                size = item.stat().st_size if item.is_file() else None
                
                contents.append({
                    "name": item.name,
                    "type": item_type,
                    "size": size,
                    "path": str(item.relative_to(base_path))
                })
            
            # Sort by type (directories first) then by name
            contents.sort(key=lambda x: (x["type"] != "directory", x["name"]))
            
            return ToolResult(
                status="success",
                output=contents,
                message=None,
                debug={"dir_path": dir_path, "item_count": len(contents)}
            )
            
        except Exception as e:
            return ToolResult(
                status="error",
                output=None,
                message=f"Failed to list directory: {str(e)}",
                debug={"dir_path": dir_path, "error_type": type(e).__name__}
            )


class DeleteFileTool(BaseTool):
    """Tool for deleting files."""
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="delete_file",
            description="Delete a file",
            parameters={
                "file_path": {
                    "type": "str",
                    "description": "Path to the file to delete",
                    "required": True
                }
            },
            examples=["delete_file(file_path='temp.txt')"],
            category="filesystem"
        )
    
    async def execute(self, file_path: str) -> ToolResult:
        """Delete a file."""
        try:
            base_path = Path(self.config.get("base_path", os.getcwd()))
            full_path = base_path / file_path
            
            # Security check
            if not str(full_path.resolve()).startswith(str(base_path.resolve())):
                return ToolResult(
                    status="error",
                    output=None,
                    message="Access denied: Path outside base directory",
                    debug={"file_path": file_path, "base_path": str(base_path)}
                )
            
            if not full_path.exists():
                return ToolResult(
                    status="error",
                    output=None,
                    message=f"File not found: {file_path}",
                    debug={"file_path": file_path, "full_path": str(full_path)}
                )
            
            if not full_path.is_file():
                return ToolResult(
                    status="error",
                    output=None,
                    message=f"Path is not a file: {file_path}",
                    debug={"file_path": file_path, "full_path": str(full_path)}
                )
            
            full_path.unlink()
            
            return ToolResult(
                status="success",
                output=f"Successfully deleted file: {file_path}",
                message=None,
                debug={"file_path": file_path}
            )
            
        except Exception as e:
            return ToolResult(
                status="error",
                output=None,
                message=f"Failed to delete file: {str(e)}",
                debug={"file_path": file_path, "error_type": type(e).__name__}
            )


class CopyFileTool(BaseTool):
    """Tool for copying files."""
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="copy_file",
            description="Copy a file to another location",
            parameters={
                "source_path": {
                    "type": "str",
                    "description": "Path to the source file",
                    "required": True
                },
                "dest_path": {
                    "type": "str", 
                    "description": "Path to the destination",
                    "required": True
                },
                "create_dirs": {
                    "type": "bool",
                    "description": "Create parent directories if they don't exist",
                    "required": False,
                    "default": True
                }
            },
            examples=["copy_file(source_path='file.txt', dest_path='backup/file.txt')"],
            category="filesystem"
        )
    
    async def execute(self, source_path: str, dest_path: str, create_dirs: bool = True) -> ToolResult:
        """Copy a file."""
        try:
            base_path = Path(self.config.get("base_path", os.getcwd()))
            full_source = base_path / source_path
            full_dest = base_path / dest_path
            
            # Security checks
            if not str(full_source.resolve()).startswith(str(base_path.resolve())):
                return ToolResult(
                    status="error",
                    output=None,
                    message="Access denied: Source path outside base directory",
                    debug={"source_path": source_path, "base_path": str(base_path)}
                )
            
            if not str(full_dest.resolve()).startswith(str(base_path.resolve())):
                return ToolResult(
                    status="error",
                    output=None,
                    message="Access denied: Destination path outside base directory",
                    debug={"dest_path": dest_path, "base_path": str(base_path)}
                )
            
            if not full_source.exists():
                return ToolResult(
                    status="error",
                    output=None,
                    message=f"Source file not found: {source_path}",
                    debug={"source_path": source_path}
                )
            
            if not full_source.is_file():
                return ToolResult(
                    status="error",
                    output=None,
                    message=f"Source is not a file: {source_path}",
                    debug={"source_path": source_path}
                )
            
            # Create parent directories if needed
            if create_dirs:
                full_dest.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(full_source, full_dest)
            
            return ToolResult(
                status="success",
                output=f"Successfully copied {source_path} to {dest_path}",
                message=None,
                debug={
                    "source_path": source_path,
                    "dest_path": dest_path,
                    "created_dirs": create_dirs
                }
            )
            
        except Exception as e:
            return ToolResult(
                status="error",
                output=None,
                message=f"Failed to copy file: {str(e)}",
                debug={
                    "source_path": source_path, 
                    "dest_path": dest_path,
                    "error_type": type(e).__name__
                }
            )


class MoveFileTool(BaseTool):
    """Tool for moving/renaming files."""
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="move_file",
            description="Move or rename a file",
            parameters={
                "source_path": {
                    "type": "str",
                    "description": "Path to the source file",
                    "required": True
                },
                "dest_path": {
                    "type": "str",
                    "description": "Path to the destination",
                    "required": True
                },
                "create_dirs": {
                    "type": "bool",
                    "description": "Create parent directories if they don't exist",
                    "required": False,
                    "default": True
                }
            },
            examples=["move_file(source_path='temp.txt', dest_path='archive/temp.txt')"],
            category="filesystem"
        )
    
    async def execute(self, source_path: str, dest_path: str, create_dirs: bool = True) -> ToolResult:
        """Move or rename a file."""
        try:
            base_path = Path(self.config.get("base_path", os.getcwd()))
            full_source = base_path / source_path
            full_dest = base_path / dest_path
            
            # Security checks
            if not str(full_source.resolve()).startswith(str(base_path.resolve())):
                return ToolResult(
                    status="error",
                    output=None,
                    message="Access denied: Source path outside base directory",
                    debug={"source_path": source_path, "base_path": str(base_path)}
                )
            
            if not str(full_dest.resolve()).startswith(str(base_path.resolve())):
                return ToolResult(
                    status="error",
                    output=None,
                    message="Access denied: Destination path outside base directory",
                    debug={"dest_path": dest_path, "base_path": str(base_path)}
                )
            
            if not full_source.exists():
                return ToolResult(
                    status="error",
                    output=None,
                    message=f"Source file not found: {source_path}",
                    debug={"source_path": source_path}
                )
            
            if not full_source.is_file():
                return ToolResult(
                    status="error",
                    output=None,
                    message=f"Source is not a file: {source_path}",
                    debug={"source_path": source_path}
                )
            
            # Create parent directories if needed
            if create_dirs:
                full_dest.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(full_source), str(full_dest))
            
            return ToolResult(
                status="success",
                output=f"Successfully moved {source_path} to {dest_path}",
                message=None,
                debug={
                    "source_path": source_path,
                    "dest_path": dest_path,
                    "created_dirs": create_dirs
                }
            )
            
        except Exception as e:
            return ToolResult(
                status="error",
                output=None,
                message=f"Failed to move file: {str(e)}",
                debug={
                    "source_path": source_path,
                    "dest_path": dest_path, 
                    "error_type": type(e).__name__
                }
            )