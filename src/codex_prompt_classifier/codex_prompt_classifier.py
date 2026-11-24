#!/usr/bin/env python3
"""codex_prompt_stats.py — 按 prompt 内容类型统计 Codex 会话日志。

用法:
  python3 codex_prompt_stats.py                          # 扫描 ~/.codex/sessions
  python3 codex_prompt_stats.py PATH [PATH ...]          # 文件/目录, .jsonl 与 .jsonl.zst 均可
  python3 codex_prompt_stats.py --top 5 ~/.codex/sessions/2026/11
  python3 codex_prompt_stats.py --json out.json PATH     # 输出 JSON 便于二次处理

分类: system_instructions / user_input / agent_output / reasoning /
      shell_command / shell_output / file_edit / other_tool_call / other_tool_output

依赖(均为可选): tiktoken 精确计数; zstandard 或 zstd CLI 读 .zst 旧日志。
无 tiktoken 时按 chars/4 估算, 无解压库时自动调用系统 zstd 命令。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("o200k_base")
except Exception:
    _ENC = None

try:
    import zstandard as zstd
except Exception:
    zstd = None

SHELL_NAMES = {"shell", "exec", "container.exec", "unified_exec", "local_shell_call"}


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text, disallowed_special=()))
    return max(1, len(text) // 4)  # 粗估: 4 字符 ≈ 1 token


def iter_lines(path: Path):
    if path.suffix == ".zst":
        if zstd is not None:
            with open(path, "rb") as fh:
                yield from zstd.ZstdDecompressor(max_window_size=2**31).stream_reader(fh).read().decode(
                    "utf-8", "replace"
                ).splitlines()
        else:
            out = subprocess.run(
                ["zstd", "-dcqf", str(path)], capture_output=True, check=True
            ).stdout.decode("utf-8", "replace")
            yield from out.splitlines()
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            yield from fh


def text_of(obj) -> str:
    """从 message content / output 字段提取纯文本。"""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):  # content parts: [{type:text|input_text|output_text, text:...}]
        parts = []
        for p in obj:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
        return "\n".join(parts)
    if isinstance(obj, dict):  # function_call_output: {"output": ...}
        if "output" in obj:
            return text_of(obj["output"])
        return json.dumps(obj, ensure_ascii=False)
    return ""


def cmd_of(arguments) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            return arguments[:200]
    if isinstance(arguments, dict):
        return str(arguments.get("cmd") or arguments.get("command") or "")[:200]
    return ""


def classify(rec: dict, call_kinds: dict) -> tuple[str, str, int]:
    """返回 (类别, 摘要, token 数)。"""
    ptype = rec.get("type")
    pl = rec.get("payload") or {}
    ptype2 = pl.get("type")

    if ptype == "session_meta":
        instr = pl.get("instructions") or []
        if isinstance(instr, str):
            instr = [instr]
        txt = "\n".join(x for x in instr if isinstance(x, str))
        return ("system_instructions", "session_meta.instructions", count_tokens(txt))

    if ptype != "response_item":
        return ("", "", 0)

    if ptype2 == "message":
        role = pl.get("role", "?")
        txt = text_of(pl.get("content"))
        cat = "user_input" if role == "user" else "agent_output"
        return (cat, txt[:60].replace("\n", " "), count_tokens(txt))

    if ptype2 == "reasoning":
        # 日志中的 reasoning 通常是加密密文, 只能统计体积并提示
        txt = text_of(pl.get("content") or pl.get("summary"))
        enc = " (encrypted)" if pl.get("encrypted_content") else ""
        return ("reasoning" + enc, "reasoning item", count_tokens(txt))

    if ptype2 in ("function_call", "custom_tool_call", "local_shell_call"):
        name = pl.get("name") or pl.get("tool_name") or "?"
        call_id = pl.get("call_id", "")
        is_shell = name in SHELL_NAMES or "cmd" in str(pl.get("arguments", ""))
        if is_shell:
            call_kinds[call_id] = "shell_output"
            return ("shell_command", cmd_of(pl.get("arguments")), count_tokens(str(pl.get("arguments", ""))))
        if "apply_patch" in name:
            call_kinds[call_id] = "file_edit_out"
            return ("file_edit", "apply_patch", count_tokens(str(pl.get("input", "") or pl.get("arguments", ""))))
        call_kinds[call_id] = "other_output"
        return ("other_tool_call", name, count_tokens(str(pl.get("arguments", pl.get("input", "")))))

    if ptype2 in ("function_call_output", "custom_tool_call_output", "local_shell_call_output"):
        call_id = pl.get("call_id", "")
        txt = text_of(pl)
        kind = call_kinds.get(call_id, "other_output")
        if kind == "shell_output":
            return ("shell_output", txt[:60].replace("\n", " "), count_tokens(txt))
        if kind == "file_edit_out":
            return ("file_edit", "patch result", count_tokens(txt))
        return ("other_tool_output", pl.get("name", "?"), count_tokens(txt))

    return ("", "", 0)


def analyze_file(path: Path):
    stats = defaultdict(lambda: {"count": 0, "tokens": 0, "chars": 0})
    big_shell = []  # (tokens, preview)
    call_kinds: dict[str, str] = {}
    codex_usage = None  # 最后一个 token_count 事件里的官方统计
    model = None
    for line in iter_lines(path):
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("type") == "turn_context" and (rec.get("payload") or {}).get("model"):
            model = rec["payload"]["model"]
        if rec.get("type") == "event_msg" and (rec.get("payload") or {}).get("type") == "token_count":
            info = (rec["payload"].get("info") or {})
            codex_usage = info.get("total_token_usage") or info.get("last_token_usage") or codex_usage
        cat, summary, toks = classify(rec, call_kinds)
        if not cat:
            continue
        plain = cat.replace(" (encrypted)", "")
        stats[plain]["count"] += 1
        stats[plain]["tokens"] += toks
        if plain == "shell_output":
            big_shell.append((toks, summary))
    return {"path": str(path), "model": model, "stats": dict(stats),
            "codex_usage": codex_usage, "top_shell": sorted(big_shell, reverse=True)[:10]}


def main():
    ap = argparse.ArgumentParser(description="按内容类型统计 Codex 会话日志")
    ap.add_argument("paths", nargs="*", default=None)
    ap.add_argument("--top", type=int, default=0, help="列出最大的 N 条 shell 输出")
    ap.add_argument("--json", metavar="OUT", help="同时写出 JSON 结果")
    args = ap.parse_args()

    roots = [Path(p).expanduser() for p in (args.paths or ["~/.codex/sessions"])]
    files = []
    for r in roots:
        if r.is_dir():
            files += sorted(r.rglob("*.jsonl")) + sorted(r.rglob("*.jsonl.zst"))
        elif r.exists():
            files.append(r)
    if not files:
        sys.exit("未找到会话文件")

    grand = defaultdict(lambda: {"count": 0, "tokens": 0})
    results = []
    for f in files:
        try:
            r = analyze_file(f)
        except Exception as e:
            print(f"[跳过] {f}: {e}", file=sys.stderr)
            continue
        results.append(r)
        for cat, s in r["stats"].items():
            grand[cat]["count"] += s["count"]
            grand[cat]["tokens"] += s["tokens"]

        total = sum(s["tokens"] for s in r["stats"].values()) or 1
        print(f"\n=== {f.name}  (model: {r['model'] or '?'}) ===")
        print(f"{'类别':<22}{'条数':>8}{'token':>12}{'占比':>8}")
        for cat, s in sorted(r["stats"].items(), key=lambda kv: -kv[1]["tokens"]):
            print(f"{cat:<22}{s['count']:>8}{s['tokens']:>12,}{s['tokens']/total*100:>7.1f}%")
        if r["codex_usage"]:
            print(f"  Codex 官方 total_token_usage: {json.dumps(r['codex_usage'], ensure_ascii=False)}")
        if args.top:
            print("  最大 shell 输出:")
            for toks, prev in r["top_shell"][: args.top]:
                print(f"    {toks:>10,} tok  {prev[:70]}")

    if len(results) > 1:
        gtotal = sum(s["tokens"] for s in grand.values()) or 1
        print("\n========== 汇总 ==========")
        for cat, s in sorted(grand.items(), key=lambda kv: -kv[1]["tokens"]):
            print(f"{cat:<22}{s['count']:>8}{s['tokens']:>12,}{s['tokens']/gtotal*100:>7.1f}%")

    if args.json:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"\nJSON 已写入 {args.json}")
    if _ENC is None:
        print("\n提示: 未安装 tiktoken, token 数为 chars/4 估算; `pip install tiktoken` 可精确计数")


if __name__ == "__main__":
    main()