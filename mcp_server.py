"""
Built-in Reference Model Context Protocol (MCP) Server.
Implements the MCP JSON-RPC 2.0 specification over Stdio.
Can run standalone or be managed automatically by the Streamlit Agent.

Tools exposed:
- get_current_datetime: Returns real-world date and time.
- eval_math: Safely computes mathematical expressions.
- system_info: Returns runtime and OS metadata.
"""

import sys
import json
import ast
import math
import operator
import platform
from datetime import datetime, timezone


def tool_get_current_datetime() -> str:
    """Returns current date and time."""
    now_local = datetime.now()
    now_utc = datetime.now(timezone.utc)
    return (
        f"Local Time: {now_local.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"UTC Time: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        f"Day of Week: {now_local.strftime('%A')}"
    )


# Operators and names the calculator accepts. Anything else is rejected.
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    name: getattr(math, name)
    for name in ("sqrt", "log", "log2", "log10", "exp", "sin", "cos", "tan",
                 "asin", "acos", "atan", "floor", "ceil", "factorial")
}
_FUNCTIONS.update({"abs": abs, "round": round, "max": max, "min": min, "pow": pow})
_CONSTANTS = {"pi": math.pi, "e": math.e}


def _eval_node(node):
    """Evaluates one node of a parsed expression, allowing only arithmetic."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS and not node.keywords):
        return _FUNCTIONS[node.func.id](*[_eval_node(arg) for arg in node.args])
    raise ValueError("only numbers, + - * / // % **, and basic math functions are allowed")


def tool_eval_math(expression: str) -> str:
    """
    Evaluates a math expression without using eval().
    The expression is parsed into a syntax tree and only whitelisted
    operators, functions and constants are executed.
    """
    if len(expression) > 200:
        return "Math Error: expression too long"
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return str(_eval_node(tree.body))
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
        if name == "get_current_datetime":
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
