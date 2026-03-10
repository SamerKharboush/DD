"""
Base Tool Class for CellType-Agent.

Provides the foundation for all tool implementations.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """
    Base class for all tools.

    Each tool must implement:
    - name: Tool identifier
    - description: What the tool does
    - run(): Execute the tool

    Usage:
        class MyTool(BaseTool):
            name = "my.tool"
            description = "Does something useful"

            def run(self, param1: str) -> dict:
                return {"result": "..."}
    """

    name: str = "base_tool"
    description: str = "Base tool class"
    parameters: dict = {}

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        pass

    def validate_params(self, **kwargs) -> bool:
        """Validate input parameters."""
        required = [k for k, v in self.parameters.items() if not k.startswith("_")]
        for param in required:
            if param not in kwargs:
                return False
        return True

    def get_schema(self) -> dict:
        """Get JSON schema for the tool."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    k: {"type": "string", "description": v}
                    for k, v in self.parameters.items()
                },
                "required": list(self.parameters.keys()),
            },
        }