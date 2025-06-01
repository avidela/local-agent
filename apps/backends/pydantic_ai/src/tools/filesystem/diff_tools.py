"""
Diff tools for PydanticAI agents.
"""

import subprocess
from pathlib import Path
from ..base import BaseTool, ToolMetadata, ToolResult


class DiffFilesTool(BaseTool):
    """Tool for comparing files using diff."""
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="diff_files",
            description="Compare two files and show differences",
            parameters={
                "file_path1": {
                    "type": "str",
                    "description": "Path to the first file",
                    "required": True
                },
                "file_path2": {
                    "type": "str",
                    "description": "Path to the second file",
                    "required": True
                },
                "unified": {
                    "type": "bool",
                    "description": "Use unified diff format",
                    "required": False,
                    "default": True
                },
                "context_lines": {
                    "type": "int",
                    "description": "Number of context lines to show",
                    "required": False,
                    "default": 3
                }
            },
            examples=["diff_files(file_path1='old.txt', file_path2='new.txt')"],
            category="filesystem"
        )
    
    async def execute(
        self,
        file_path1: str,
        file_path2: str,
        unified: bool = True,
        context_lines: int = 3
    ) -> ToolResult:
        """Compare two files and show differences."""
        try:
            base_path = Path(self.config.get("base_path", "."))
            full_path1 = base_path / file_path1
            full_path2 = base_path / file_path2
            
            # Security checks
            if not str(full_path1.resolve()).startswith(str(base_path.resolve())):
                return ToolResult(
                    status="error",
                    output=None,
                    message="Access denied: First file path outside base directory",
                    debug={"file_path1": file_path1, "base_path": str(base_path)}
                )
            
            if not str(full_path2.resolve()).startswith(str(base_path.resolve())):
                return ToolResult(
                    status="error",
                    output=None,
                    message="Access denied: Second file path outside base directory", 
                    debug={"file_path2": file_path2, "base_path": str(base_path)}
                )
            
            if not full_path1.exists():
                return ToolResult(
                    status="error",
                    output=None,
                    message=f"First file not found: {file_path1}",
                    debug={"file_path1": file_path1}
                )
            
            if not full_path2.exists():
                return ToolResult(
                    status="error",
                    output=None,
                    message=f"Second file not found: {file_path2}",
                    debug={"file_path2": file_path2}
                )
            
            if not full_path1.is_file():
                return ToolResult(
                    status="error",
                    output=None,
                    message=f"First path is not a file: {file_path1}",
                    debug={"file_path1": file_path1}
                )
            
            if not full_path2.is_file():
                return ToolResult(
                    status="error",
                    output=None,
                    message=f"Second path is not a file: {file_path2}",
                    debug={"file_path2": file_path2}
                )
            
            # Build diff command
            cmd = ["diff"]
            if unified:
                cmd.extend(["-u", f"-{context_lines}"])
            
            cmd.extend([str(full_path1), str(full_path2)])
            
            # Execute diff command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=base_path
            )
            
            if result.returncode == 0:
                # Files are identical
                return ToolResult(
                    status="success",
                    output="Files are identical.",
                    message=None,
                    debug={
                        "file_path1": file_path1,
                        "file_path2": file_path2,
                        "identical": True,
                        "command": " ".join(cmd)
                    }
                )
            elif result.returncode == 1:
                # Files differ
                diff_output = result.stdout.strip()
                return ToolResult(
                    status="success",
                    output=diff_output,
                    message=None,
                    debug={
                        "file_path1": file_path1,
                        "file_path2": file_path2,
                        "identical": False,
                        "diff_lines": len(diff_output.split('\n')) if diff_output else 0,
                        "command": " ".join(cmd)
                    }
                )
            else:
                # Error occurred
                return ToolResult(
                    status="error",
                    output=None,
                    message=f"Diff command failed: {result.stderr.strip()}",
                    debug={
                        "file_path1": file_path1,
                        "file_path2": file_path2,
                        "command": " ".join(cmd),
                        "return_code": result.returncode,
                        "stderr": result.stderr.strip()
                    }
                )
                
        except Exception as e:
            return ToolResult(
                status="error",
                output=None,
                message=f"Diff operation failed: {str(e)}",
                debug={
                    "file_path1": file_path1,
                    "file_path2": file_path2,
                    "error_type": type(e).__name__
                }
            )