"""
Calculator tool for PydanticAI agents.
"""

import ast
import operator
from typing import Any, Dict
from ..base import BaseTool, ToolMetadata, ToolResult


class CalculatorTool(BaseTool):
    """Tool for performing mathematical calculations."""
    
    # Supported operations
    _operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    # Supported functions
    _functions = {
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'pow': pow,
    }
    
    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="calculator",
            description="Perform mathematical calculations",
            parameters={
                "expression": {
                    "type": "str",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4', 'pow(2, 8)')",
                    "required": True
                }
            },
            examples=[
                "calculator(expression='2 + 3 * 4')",
                "calculator(expression='pow(2, 8)')",
                "calculator(expression='15 * 23 + 47')"
            ],
            category="computation"
        )
    
    def _evaluate_node(self, node):
        """Safely evaluate an AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # For Python < 3.8 compatibility
            return node.n
        elif isinstance(node, ast.BinOp):
            left = self._evaluate_node(node.left)
            right = self._evaluate_node(node.right)
            op = self._operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._evaluate_node(node.operand)
            op = self._operators.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operation: {type(node.op).__name__}")
            return op(operand)
        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else None
            if func_name not in self._functions:
                raise ValueError(f"Unsupported function: {func_name}")
            
            args = [self._evaluate_node(arg) for arg in node.args]
            return self._functions[func_name](*args)
        elif isinstance(node, ast.Name):
            # Only allow specific constants
            if node.id == 'pi':
                return 3.141592653589793
            elif node.id == 'e':
                return 2.718281828459045
            else:
                raise ValueError(f"Unsupported variable: {node.id}")
        else:
            raise ValueError(f"Unsupported node type: {type(node).__name__}")
    
    async def execute(self, expression: str) -> ToolResult:
        """Execute mathematical calculation."""
        try:
            # Parse the expression into an AST
            tree = ast.parse(expression.strip(), mode='eval')
            
            # Evaluate the expression safely
            result = self._evaluate_node(tree.body)
            
            # Format the result
            if isinstance(result, float):
                # Round to reasonable precision for display
                if result.is_integer():
                    result = int(result)
                else:
                    result = round(result, 10)
            
            return ToolResult(
                status="success",
                output=str(result),
                message=None,
                debug={
                    "expression": expression,
                    "result": result,
                    "result_type": type(result).__name__
                }
            )
            
        except SyntaxError as e:
            return ToolResult(
                status="error",
                output=None,
                message=f"Invalid mathematical expression: {str(e)}",
                debug={
                    "expression": expression,
                    "error_type": "SyntaxError"
                }
            )
        except ValueError as e:
            return ToolResult(
                status="error",
                output=None,
                message=f"Calculation error: {str(e)}",
                debug={
                    "expression": expression,
                    "error_type": "ValueError"
                }
            )
        except ZeroDivisionError:
            return ToolResult(
                status="error",
                output=None,
                message="Division by zero",
                debug={
                    "expression": expression,
                    "error_type": "ZeroDivisionError"
                }
            )
        except Exception as e:
            return ToolResult(
                status="error",
                output=None,
                message=f"Calculation failed: {str(e)}",
                debug={
                    "expression": expression,
                    "error_type": type(e).__name__
                }
            )