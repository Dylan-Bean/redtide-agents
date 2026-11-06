"""Teaching step 4: speak the MCP protocol to the server by hand — no
ready-made client involved.

Run (from the repo root): py scripts/04_test_mcp.py

MCP in essence = JSON-RPC 2.0 over stdio:
    initialize (handshake: protocol version + capabilities)
    -> notifications/initialized (acknowledge)
    -> tools/list (client asks which tools exist)
    -> tools/call (the actual call; arguments follow JSON Schema)
Once these four messages make sense, MCP is no longer a black box — this is
exactly what Claude Code / Desktop do when they connect.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

proc = subprocess.Popen(
    [sys.executable, str(ROOT / "mcp" / "redtide_mcp_server.py")],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL, text=True, encoding="utf-8", cwd=ROOT)
print("[*] server 子进程已启动（首次加载嵌入模型需十几秒）", flush=True)


def send(obj):
    proc.stdin.write(json.dumps(obj) + "\n")
    proc.stdin.flush()


def recv():
    line = proc.stdout.readline()
    return json.loads(line)


try:
    send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
          "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                     "clientInfo": {"name": "manual-test", "version": "0.1"}}})
    init = recv()
    print("1) initialize ->", init["result"]["serverInfo"], flush=True)

    send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = recv()["result"]["tools"]
    print("2) tools/list ->", [t["name"] for t in tools], flush=True)

    send({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
          "params": {"name": "kb_search",
                     "arguments": {"query": "夜光藻 有毒吗", "top_k": 2}}})
    out = recv()["result"]["content"][0]["text"]
    print("3) tools/call kb_search ->", out[:260], "…", flush=True)

    send({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
          "params": {"name": "check_warning_threshold",
                     "arguments": {"species": "夜光藻", "cell_density": 50000}}})
    out = recv()["result"]["content"][0]["text"]
    print("4) tools/call check_warning_threshold ->", out, flush=True)

    print("\nMCP 协议自测通过 ✓")
finally:
    proc.terminate()
