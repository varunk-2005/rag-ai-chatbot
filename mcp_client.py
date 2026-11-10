"""
Model Context Protocol (MCP) Client Implementation.
Conforms to the Anthropic MCP JSON-RPC 2.0 specification over Stdio transport.
Provides:
- StdioMCPClient: Communicates with an individual MCP server process.
- MCPServerManager: Manages multiple MCP server instances, auto-discovers tools,
  and translates schemas for LLM Agent function calling.
"""

import json
import select
import subprocess
import threading
import sys
import os
from typing import Dict, List, Any, Optional


class StdioMCPClient:
    """
    Client for a single MCP server communicating via standard input/output (stdio)
    using JSON-RPC 2.0 protocol.
    """

    def __init__(self, name: str, command: str, args: List[str], env: Optional[Dict[str, str]] = None, cwd: Optional[str] = None):
        self.name = name
        self.command = command
        self.args = args
        self.env = env or os.environ.copy()
        self.cwd = cwd or os.getcwd()
        self.process: Optional[subprocess.Popen] = None
        self.request_id = 0
        self.tools: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self.is_connected = False

    def connect(self) -> bool:
        """Starts the server subprocess and performs MCP initialization handshake."""
        try:
            full_cmd = [self.command] + self.args
            self.process = subprocess.Popen(
                full_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                # Discard server stderr: an unread pipe can fill up and block the server
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                cwd=self.cwd,
                env=self.env
            )

            # 1. MCP Initialize Request
            init_response = self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "rag-ai-chatbot-agent",
                    "version": "2.0.0"
                }
            })

            if not init_response or "error" in init_response:
                self.disconnect()
                return False

            # 2. MCP Initialized Notification
            self._send_notification("notifications/initialized", {})

            # 3. Discover available tools
            self.tools = self.list_tools()
            self.is_connected = True
            return True
        except Exception as e:
            self.disconnect()
            return False

    def disconnect(self):
        """Terminates the server subprocess cleanly."""
        self.is_connected = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None

    def _send_request(self, method: str, params: Optional[Dict[str, Any]] = None, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
        """Sends a JSON-RPC 2.0 request and waits synchronously for the response line."""
        if not self.process or self.process.poll() is not None:
            return None

        with self._lock:
            self.request_id += 1
            req_id = self.request_id
            payload = {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params or {}
            }
            try:
                json_str = json.dumps(payload) + "\n"
                self.process.stdin.write(json_str)
                self.process.stdin.flush()

                # Read response line, but never wait longer than `timeout` seconds,
                # so a stuck server cannot freeze the app
                ready, _, _ = select.select([self.process.stdout], [], [], timeout)
                if not ready:
                    return None
                raw_line = self.process.stdout.readline()
                if not raw_line:
                    return None
                response = json.loads(raw_line.strip())
                return response
            except Exception:
                return None

    def _send_notification(self, method: str, params: Optional[Dict[str, Any]] = None):
        """Sends a JSON-RPC 2.0 notification (no response expected)."""
        if not self.process or self.process.poll() is not None:
            return

        with self._lock:
            payload = {
                "jsonrpc": "2.0",
                "method": method,
                "params": params or {}
            }
            try:
                json_str = json.dumps(payload) + "\n"
                self.process.stdin.write(json_str)
                self.process.stdin.flush()
            except Exception:
                pass

    def list_tools(self) -> List[Dict[str, Any]]:
        """Discovers tools supported by this MCP server."""
        resp = self._send_request("tools/list", {})
        if resp and "result" in resp and "tools" in resp["result"]:
            self.tools = resp["result"]["tools"]
            return self.tools
        return []

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Calls a tool on this MCP server and returns the output text."""
        resp = self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })

        if not resp:
            return "Error: MCP server did not respond."
        if "error" in resp:
            return f"MCP Error: {resp['error'].get('message', 'Unknown error')}"

        result = resp.get("result", {})
        content = result.get("content", [])
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(item.get("text", ""))
                elif isinstance(item, str):
                    texts.append(item)
            return "\n".join(texts) if texts else str(result)
        return str(result)


class MCPServerManager:
    """
    Manages connections to multiple MCP servers, registers their tools,
    and handles tool routing from LLM agents.
    """

    def __init__(self, config_file: Optional[str] = None):
        self.clients: Dict[str, StdioMCPClient] = {}
        self.tool_to_server_map: Dict[str, str] = {}
        self.config_file = config_file or "mcp_servers.json"

    def register_server(self, name: str, command: str, args: List[str], cwd: Optional[str] = None, auto_connect: bool = True) -> bool:
        """Registers and connects an MCP server."""
        if name in self.clients:
            self.clients[name].disconnect()

        client = StdioMCPClient(name=name, command=command, args=args, cwd=cwd)
        if auto_connect:
            success = client.connect()
            if success:
                self.clients[name] = client
                self._update_tool_registry()
                return True
            return False
        else:
            self.clients[name] = client
            return True

    def _update_tool_registry(self):
        """Refreshes the mapping from tool name to server client."""
        self.tool_to_server_map.clear()
        for server_name, client in self.clients.items():
            if client.is_connected:
                for tool in client.tools:
                    tool_name = tool.get("name")
                    if tool_name:
                        self.tool_to_server_map[tool_name] = server_name

    def load_from_config(self, filepath: Optional[str] = None):
        """Loads MCP servers configuration from a JSON file."""
        cfg_path = filepath or self.config_file
        if not os.path.exists(cfg_path):
            return

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            servers = data.get("mcpServers", {})
            for name, spec in servers.items():
                cmd = spec.get("command", sys.executable)
                args = spec.get("args", [])
                cwd = spec.get("cwd")
                self.register_server(name, cmd, args, cwd=cwd, auto_connect=True)
        except Exception:
            pass

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Returns all tools from all connected MCP servers."""
        all_tools = []
        for server_name, client in self.clients.items():
            if client.is_connected:
                for tool in client.tools:
                    tool_copy = dict(tool)
                    tool_copy["server_name"] = server_name
                    all_tools.append(tool_copy)
        return all_tools

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Routes a tool call to the appropriate MCP server."""
        server_name = self.tool_to_server_map.get(tool_name)
        if not server_name or server_name not in self.clients:
            return f"Error: No connected MCP server provides tool '{tool_name}'."

        client = self.clients[server_name]
        return client.call_tool(tool_name, arguments)

    def shutdown_all(self):
        """Disconnects all MCP servers."""
        for client in self.clients.values():
            client.disconnect()
        self.clients.clear()
        self.tool_to_server_map.clear()
