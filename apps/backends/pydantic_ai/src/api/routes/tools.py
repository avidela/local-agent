"""
Tools API routes for PydanticAI agents.
"""

from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ...services.tools.tool_service import get_tool_service
from ...tools.base import ToolMetadata

router = APIRouter()


class ToolExecuteRequest(BaseModel):
    """Request schema for tool execution."""
    tool_name: str
    parameters: Dict[str, Any] = {}
    config: Optional[Dict[str, Any]] = None


class ToolExecuteResponse(BaseModel):
    """Response schema for tool execution."""
    status: str
    output: Optional[Any] = None
    message: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None


@router.get("/available", response_model=List[str])
async def get_available_tools(
    category: Optional[str] = None,
    enabled_only: bool = True
):
    """
    Get list of available tool names.
    
    Args:
        category: Filter by tool category
        enabled_only: Only return enabled tools
        
    Returns:
        List of available tool names
    """
    tool_service = get_tool_service()
    return tool_service.get_available_tools(category=category, enabled_only=enabled_only)


@router.get("/metadata", response_model=Dict[str, ToolMetadata])
async def get_all_tool_metadata(category: Optional[str] = None):
    """
    Get metadata for all tools.
    
    Args:
        category: Filter by tool category
        
    Returns:
        Dictionary of tool metadata
    """
    tool_service = get_tool_service()
    return tool_service.get_all_tool_metadata(category=category)


@router.get("/{tool_name}/metadata", response_model=ToolMetadata)
async def get_tool_metadata(tool_name: str):
    """
    Get metadata for a specific tool.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Tool metadata
    """
    tool_service = get_tool_service()
    metadata = tool_service.get_tool_metadata(tool_name)
    
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found"
        )
    
    return metadata


@router.post("/execute", response_model=ToolExecuteResponse)
async def execute_tool(request: ToolExecuteRequest):
    """
    Execute a tool with given parameters.
    
    Args:
        request: Tool execution request
        
    Returns:
        Tool execution result
    """
    tool_service = get_tool_service()
    
    # Check if tool exists
    metadata = tool_service.get_tool_metadata(request.tool_name)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{request.tool_name}' not found"
        )
    
    # Execute the tool
    result = await tool_service.execute_tool(
        tool_name=request.tool_name,
        config=request.config,
        **request.parameters
    )
    
    return ToolExecuteResponse(
        status=result.status,
        output=result.output,
        message=result.message,
        debug=result.debug
    )


@router.get("/categories", response_model=List[str])
async def get_tool_categories():
    """
    Get list of tool categories.
    
    Returns:
        List of tool categories
    """
    tool_service = get_tool_service()
    all_metadata = tool_service.get_all_tool_metadata()
    
    categories = set()
    for metadata in all_metadata.values():
        categories.add(metadata.category)
    
    return sorted(list(categories))


@router.get("/by-category/{category}", response_model=List[str])
async def get_tools_by_category(category: str, enabled_only: bool = True):
    """
    Get tools by category.
    
    Args:
        category: Tool category
        enabled_only: Only return enabled tools
        
    Returns:
        List of tool names in the category
    """
    tool_service = get_tool_service()
    return tool_service.get_available_tools(category=category, enabled_only=enabled_only)


@router.get("/{tool_name}/schema", response_model=Dict[str, Any])
async def get_tool_schema(tool_name: str):
    """
    Get tool schema in PydanticAI format.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Tool schema for PydanticAI
    """
    tool_service = get_tool_service()
    metadata = tool_service.get_tool_metadata(tool_name)
    
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found"
        )
    
    schemas = tool_service.get_tool_schemas_for_pydantic_ai([tool_name])
    return schemas[0] if schemas else {}


@router.post("/{tool_name}/test", response_model=ToolExecuteResponse)
async def test_tool(tool_name: str, parameters: Dict[str, Any] = {}):
    """
    Test a tool with sample parameters.
    
    Args:
        tool_name: Name of the tool to test
        parameters: Test parameters
        
    Returns:
        Tool execution result
    """
    tool_service = get_tool_service()
    
    # Check if tool exists
    metadata = tool_service.get_tool_metadata(tool_name)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found"
        )
    
    # Execute the tool with test config
    test_config = {"base_path": "/tmp/test"}  # Safe test environment
    result = await tool_service.execute_tool(
        tool_name=tool_name,
        config=test_config,
        **parameters
    )
    
    return ToolExecuteResponse(
        status=result.status,
        output=result.output,
        message=result.message,
        debug=result.debug
    )