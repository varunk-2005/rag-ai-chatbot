"""
Built-in Reference Model Context Protocol (MCP) Server.
Implements the MCP JSON-RPC 2.0 specification over Stdio.
Can run standalone or be managed automatically by the Streamlit Agent.

Tools exposed:
- fetch_web_url: Fetches and cleans textual content from a webpage.
- get_current_datetime: Returns real-world date and time.
- eval_math: Safely computes mathematical expressions.
- system_info: Returns runtime and OS metadata.
"""

import sys
import json
import math
import platform
import urllib.request
import re
from datetime import datetime, timezone


def clean_html(raw_html: str) -> str:
    """Strips script, style tags and converts HTML to clean readable text."""
    text = re.sub(r'<(script|style)\b[^>]*>[\s\S]*?<\/\1>', '', raw_html, flags=re.IGNORECASE)
    text = re.sub(r'<(br|p|div|li|tr)[^>]*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()


def tool_fetch_web_url(url: str, max_chars: int = 4000) -> str:
    """Fetches a URL and returns clean readable text."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; RAG-Agent-MCP/1.0)"}
        )
        with urllib.request.urlopen(req, timeout=8) as response:
            html = response.read().decode('utf-8', errors='replace')
            cleaned = clean_html(html)
            if len(cleaned) > max_chars:
                cleaned = cleaned[:max_chars] + f"\n... [Truncated to {max_chars} chars]"
            return f"Source URL: {url}\n\nContent:\n{cleaned}"
    except Exception as e:
        return f"Failed to fetch URL '{url}': {str(e)}"


def tool_get_current_datetime() -> str:
    """Returns current date and time."""
    now_local = datetime.now()
    now_utc = datetime.now(timezone.utc)
    return (
        f"Local Time: {now_local.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"UTC Time: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        f"Day of Week: {now_local.strftime('%A')}"
    )


def tool_eval_math(expression: str) -> str:
    """Safely evaluates a math expression using math library functions."""
    safe_scope = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
    safe_scope["math"] = math
    safe_scope["abs"] = abs
    safe_scope["round"] = round
    safe_scope["max"] = max
    safe_scope["min"] = min
    safe_scope["pow"] = pow
    safe_scope["sum"] = sum

    # Disallow dangerous tokens
    for token in ["__", "import", "open", "exec", "eval", "os", "sys", "subprocess"]:
        if token in expression:
            return f"Error: Expression contains forbidden keyword '{token}'"

    try:
        result = eval(expression, {"__builtins__": {}}, safe_scope)
        return str(result)
    except Exception as e:
        return f"Math Error: {str(e)}"


def tool_system_info() -> str:
    """Returns environment information."""
    return (
        f"Platform: {platform.system()} {platform.release()}\n"
        f"Machine: {platform.machine()}\n"
        f"Python Version: {platform.python_version()}"
    )


# MCP Tool Definitions (JSON Schema)
MCP_TOOLS = [
    {
        "name": "fetch_web_url",
        "description": "Fetches external content from a webpage URL and returns readable text.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The full HTTP/HTTPS URL to fetch."}
            },
            "required": ["url"]
        }
    },
    {
        "name": "get_current_datetime",
        "description": "Returns the current real-world date, time, weekday, and UTC time.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "eval_math",
        "description": "Safely evaluates mathematical expressions (supports sqrt, log, sin, cos, basic arithmetic).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Mathematical formula to evaluate, e.g. '(150-120)/120 * 100' or 'sqrt(144)'."}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "system_info",
        "description": "Returns runtime environment information including OS platform and Python version.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }
]


def handle_request(req: dict) -> dict:
    """Handles an MCP JSON-RPC 2.0 request."""
    method = req.get("method")
    req_id = req.get("id")
    params = req.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "rag-builtin-mcp-server",
                    "version": "1.0.0"
                }
            }
        }

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": MCP_TOOLS
            }
        }

    elif method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {})

        output = ""
        if name == "fetch_web_url":
            output = tool_fetch_web_url(args.get("url", ""))
        elif name == "get_current_datetime":
            output = tool_get_current_datetime()
        elif name == "eval_math":
            output = tool_eval_math(args.get("expression", ""))
        elif name == "system_info":
            output = tool_system_info()
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Tool '{name}' not found"
                }
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": output
                    }
                ]
            }
        }

    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32601,
                "message": f"Method '{method}' not found"
            }
        }


def main():
    """Main stdio loop for the MCP server."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if "id" not in req:
                continue
            response = handle_request(req)
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {str(e)}"
                }
            }
            sys.stdout.write(json.dumps(err_resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
