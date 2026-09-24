"""MCP tool modules registered onto the server at start-up.

Each module exposes TOOL_NAME, ACTIONS, POLICY and register(mcp, ctx). See
mcp_tools/io_tools.py for the contract.
"""

from . import config_tools, io_tools

MODULES = (io_tools, config_tools)

__all__ = ["MODULES", "io_tools", "config_tools"]
