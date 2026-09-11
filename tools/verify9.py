#!/usr/bin/env python3
"""Boot 9 checks (run on Reddie): count-to-100 speed, a tool call, then one tiny image (last: FlashInfer #4973)."""
import base64, json, struct, time, urllib.request, zlib

U = "http://127.0.0.1:8000/v1/chat/completions"
M = "deepseek-v4.1-flash"


def post(body, timeout=600):
    req = urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    return r, time.time() - t


for run in (1, 2):
    r, dt = post({"model": M, "messages": [{"role": "user", "content": "Count from 1 to 100, separated by spaces. Output only the numbers."}],
                  "max_tokens": 400, "temperature": 0})
    ct = r["usage"]["completion_tokens"]
    nums = [int(x) for x in (r["choices"][0]["message"]["content"] or "").split() if x.isdigit()]
    print(f"count run{run}: {ct} tok / {dt:.2f}s = {ct / dt:.1f} tok/s | correct 1..100: {nums[:100] == list(range(1, 101))}", flush=True)

CODE = ("Write a Python function merge_intervals(intervals) that merges overlapping intervals and returns them sorted. "
        "Include a one-line docstring and two example calls.")
for run in (1, 2):
    r, dt = post({"model": M, "messages": [{"role": "user", "content": f"[code {run}] " + CODE}], "max_tokens": 200, "temperature": 0})
    ct = r["usage"]["completion_tokens"]
    print(f"code run{run} (bench v1 coding prompt, non-streaming): {ct} tok / {dt:.2f}s = {ct / dt:.1f} tok/s", flush=True)

tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                         "required": ["city"]}}}]
r, dt = post({"model": M, "messages": [{"role": "user", "content": "What's the weather in Paris right now, in celsius?"}],
              "tools": tools, "tool_choice": "auto", "max_tokens": 200, "temperature": 0})
ch = r["choices"][0]
calls = [(c["function"]["name"], c["function"]["arguments"]) for c in (ch["message"].get("tool_calls") or [])]
print(f"tool call: finish_reason={ch['finish_reason']} tool_calls={calls} content={ch['message'].get('content')!r} ({dt:.1f}s)", flush=True)

w = h = 64
raw = b"".join(b"\x00" + b"\xff\x00\x00" * w for _ in range(h))


def chunk(t, d):
    return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)


png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
r, dt = post({"model": M, "max_tokens": 20, "temperature": 0, "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What single color fills this image? Answer with one word."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}]}]})
print(f"vision: answer={r['choices'][0]['message'].get('content')!r} prompt_tokens={r['usage']['prompt_tokens']} ({dt:.1f}s)", flush=True)
