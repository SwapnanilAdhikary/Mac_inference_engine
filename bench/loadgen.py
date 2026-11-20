"""Closed-loop load generator for OpenAI-compatible streaming servers (stdlib asyncio HTTP).

For each concurrency level C, C workers issue requests back to back until
--num-requests have completed. Reports TTFT, inter-token latency (gaps between
SSE chunks that carry text), TPOT, end-to-end latency, output tok/s and req/s.

    HF_HUB_OFFLINE=1 uv run python bench/loadgen.py --url http://127.0.0.1:8101 \
        --model Qwen/Qwen3-0.6B --workload chat --concurrency 1,2,4,8,16 \
        --num-requests 64 --out results.json

Prompts are synthetic and seeded, so every engine gets byte-identical requests.
Each level uses its own prompts (and its own shared prefix for `prefix`), so a
prefix cache warmed by one level cannot help the next.
"""

import argparse
import asyncio
import json
import random
import time
from collections import Counter
from urllib.parse import urlsplit

QWEN3_EOS_IDS = (151643, 151645)  # <|endoftext|>, <|im_end|>

WORDS = """the of and to in is was for on that with as by at from his her it an were are
which this be or has had not but also its their they after first new two one
year city time world state during more other system river music team water
light house school north early large small often power field market model
data network memory engine signal value number process energy space group
second order human local public history language island design paper road
village church army war film game party song book station island bridge
mountain valley forest garden ocean winter summer stone metal glass wind
fire cloud morning evening night window letter story voice color animal""".split()


# ---------------------------------------------------------------- prompts

def synth_text(rng, tok, n_tokens):
    """Seeded pseudo-English text that encodes to exactly ~n_tokens tokens."""
    words = []
    while True:
        for _ in range(n_tokens):
            w = rng.choice(WORDS)
            if not words or words[-1].endswith("."):
                w = w.capitalize()
            words.append(w + ("." if rng.random() < 0.08 else ""))
        ids = tok.encode(" ".join(words), add_special_tokens=False)
        if len(ids) >= n_tokens:
            return tok.decode(ids[:n_tokens])


def make_requests(workload, n, max_tokens, tok, seed):
    """List of (system_or_None, user_text, max_tokens)."""
    rng = random.Random(seed)
    if workload == "chat":
        return [(None, synth_text(rng, tok, rng.randint(100, 300)),
                 max_tokens or rng.randint(128, 256)) for _ in range(n)]
    if workload == "long":
        return [(None, synth_text(rng, tok, rng.randint(2048, 4096)), max_tokens or 64)
                for _ in range(n)]
    if workload == "prefix":
        prefix = "You are a helpful assistant. Reference document:\n" + synth_text(rng, tok, 2048)
        return [(prefix, f"Request {i}: " + synth_text(rng, tok, rng.randint(32, 64)),
                 max_tokens or 64) for i in range(n)]
    raise ValueError(workload)


def build_body(req, args):
    system, user, max_tokens = req
    body = {"model": args.model, "stream": True, "temperature": 0, "max_tokens": max_tokens,
            "ignore_eos": True, "stream_options": {"include_usage": True}}
    if args.suppress_eos:  # for engines that ignore `ignore_eos`
        body["logit_bias"] = {str(t): -100 for t in QWEN3_EOS_IDS}
    if args.endpoint == "chat":
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        body["messages"] = msgs
    else:
        body["prompt"] = (system + "\n\n" if system else "") + user
    return body


def count_prompt_tokens(body, tok):
    if "messages" in body:
        text = tok.apply_chat_template(body["messages"], tokenize=False, add_generation_prompt=True)
        return len(tok.encode(text, add_special_tokens=False))
    return len(tok.encode(body["prompt"], add_special_tokens=False))


# ---------------------------------------------------------------- HTTP + SSE

async def iter_body(reader, chunked):
    if chunked:
        while True:
            size = int((await reader.readline()).split(b";")[0], 16)
            if size == 0:
                return
            data = await reader.readexactly(size)
            await reader.readexactly(2)  # CRLF
            yield data
    else:  # Connection: close -> body ends at EOF
        while data := await reader.read(65536):
            yield data


async def stream_request(host, port, path, body):
    """POST `body`, parse the SSE stream, return a per-request record."""
    rec = {"ok": False, "text": "", "token_times": [], "finish_reason": None, "usage": None}
    payload = json.dumps(body).encode()
    t0 = time.perf_counter()
    reader, writer = await asyncio.open_connection(host, port, limit=1 << 22)
    try:
        writer.write(
            f"POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\nContent-Type: application/json\r\n"
            f"Accept: text/event-stream\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode()
            + payload)
        await writer.drain()
        status = int((await reader.readline()).split()[1])
        headers = {}
        while (line := await reader.readline()) not in (b"\r\n", b"\n", b""):
            k, _, v = line.decode("latin-1").partition(":")
            headers[k.strip().lower()] = v.strip().lower()
        chunked = "chunked" in headers.get("transfer-encoding", "")
        if status != 200:
            err = b"".join([d async for d in iter_body(reader, chunked)])
            raise RuntimeError(f"HTTP {status}: {err[:300].decode(errors='replace')}")
        buf, done, text = b"", False, []
        async for data in iter_body(reader, chunked):
            now = time.perf_counter()
            buf += data
            *lines, buf = buf.split(b"\n")
            for line in lines:
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                line = line[5:].strip()
                if line == b"[DONE]":
                    done = True
                    break
                ev = json.loads(line)
                if ev.get("error"):
                    raise RuntimeError(f"server error: {ev['error']}")
                if ev.get("usage"):
                    rec["usage"] = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    piece = (ch.get("text") or "") + (d.get("content") or "") + \
                        (d.get("reasoning_content") or "") + (d.get("reasoning") or "")
                    if piece:
                        text.append(piece)
                        rec["token_times"].append(now - t0)
                    if ch.get("finish_reason"):
                        rec["finish_reason"] = ch["finish_reason"]
            if done:
                break
        rec["e2e"] = time.perf_counter() - t0
        rec["text"] = "".join(text)
        rec["ok"] = bool(rec["token_times"])
        if not rec["ok"]:
            rec["error"] = "no tokens received"
    finally:
        writer.close()
    return rec


# ---------------------------------------------------------------- stats

def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def dist(xs, scale=1000.0):
    """p50/p90/p99/mean of seconds, reported in ms."""
    if not xs:
        return {"p50": None, "p90": None, "p99": None, "mean": None}
    return {"p50": pct(xs, 50) * scale, "p90": pct(xs, 90) * scale,
            "p99": pct(xs, 99) * scale, "mean": sum(xs) / len(xs) * scale}


def summarize(level_recs, reqs_meta, wall, conc):
    ok = [r for r in level_recs if r["ok"]]
    ttft = [r["token_times"][0] for r in ok]
    itl = [b - a for r in ok for a, b in zip(r["token_times"], r["token_times"][1:])]
    tpot = [(r["e2e"] - r["token_times"][0]) / (r["out_tokens"] - 1) for r in ok if r["out_tokens"] > 1]
    e2e = [r["e2e"] for r in ok]
    total_out = sum(r["out_tokens"] for r in ok)
    mismatched = [r for r in ok if r["out_tokens"] != r["max_tokens"]]
    return {
        "concurrency": conc,
        "num_requests": len(level_recs),
        "completed": len(ok),
        "failed": len(level_recs) - len(ok),
        "errors": Counter(r.get("error", "?") for r in level_recs if not r["ok"]).most_common(3),
        "wall_s": wall,
        "req_per_s": len(ok) / wall,
        "output_tok_per_s": total_out / wall,
        "ttft_ms": dist(ttft),
        "itl_ms": dist(itl),
        "tpot_ms": dist(tpot),
        "e2e_ms": dist(e2e),
        "prompt_tokens_mean": sum(m["prompt_tokens"] for m in reqs_meta) / len(reqs_meta),
        "server_prompt_tokens_mean": _mean([r["usage"].get("prompt_tokens") for r in ok if r["usage"]]),
        "output_tokens_mean": total_out / len(ok) if ok else None,
        "retokenized_output_tokens_mean": _mean([r["retok_tokens"] for r in ok]),
        "chunks_per_request_mean": _mean([len(r["token_times"]) for r in ok]),
        "usage_reported": sum(1 for r in ok if r["usage"]),
        "length_mismatches": len(mismatched),
        "finish_reasons": dict(Counter(r["finish_reason"] for r in ok)),
    }


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


# ---------------------------------------------------------------- driver

async def run_level(host, port, path, bodies, conc):
    recs = [None] * len(bodies)
    todo = iter(range(len(bodies)))

    async def worker():
        for i in todo:  # shared iterator: each worker pulls the next request
            try:
                recs[i] = await stream_request(host, port, path, bodies[i])
            except Exception as e:  # record and keep going
                recs[i] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300], "token_times": []}

    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(conc)))
    return recs, time.perf_counter() - t0


def trio(d, nd=1, scale=1.0, keys=("p50", "p90", "p99")):
    return "/".join("-" if d[k] is None else f"{d[k] * scale:.{nd}f}" for k in keys)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="server base URL")
    ap.add_argument("--model", required=True, help="model name sent in the request")
    ap.add_argument("--endpoint", choices=["completions", "chat"], default="completions")
    ap.add_argument("--concurrency", default="1,2,4,8,16")
    ap.add_argument("--num-requests", type=int, default=64, help="requests per concurrency level")
    ap.add_argument("--workload", choices=["chat", "long", "prefix"], default="chat")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="fixed output length (default: chat 128-256 seeded, long/prefix 64)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suppress-eos", action="store_true",
                    help="also send logit_bias=-100 on Qwen3 EOS ids (for engines without ignore_eos)")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args()

    from transformers import AutoTokenizer  # slow import; keep --help fast

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    u = urlsplit(args.url)
    host, port = u.hostname, u.port or 80
    base = u.path.rstrip("/").removesuffix("/v1")
    path = base + ("/v1/chat/completions" if args.endpoint == "chat" else "/v1/completions")
    levels = [int(c) for c in args.concurrency.split(",")]

    if not args.no_warmup:
        body = build_body(make_requests("chat", 1, 16, tok, f"{args.seed}-warmup")[0], args)
        rec = asyncio.run(stream_request(host, port, path, body))
        print(f"warmup: ok={rec['ok']} e2e={rec.get('e2e', 0):.2f}s {rec.get('error', '')}")

    print(f"{args.workload} workload, {args.endpoint} endpoint, {args.num_requests} req/level -> {args.url}")
    print(f"{'conc':>4} {'ok':>6} {'wall_s':>7} {'req/s':>6} {'out_tok/s':>9} "
          f"{'TTFT p50/p90/p99 ms':>22} {'ITL p50/p90/p99 ms':>20} {'TPOT p50':>8} "
          f"{'E2E p50/p99 s':>14} {'in_tok':>6} {'out_tok':>7} {'len_bad':>7}")
    results = []
    all_mismatch = 0
    for conc in levels:
        reqs = make_requests(args.workload, args.num_requests, args.max_tokens, tok,
                             f"{args.seed}-{args.workload}-{conc}")
        bodies = [build_body(r, args) for r in reqs]
        meta = [{"prompt_tokens": count_prompt_tokens(b, tok), "max_tokens": b["max_tokens"]} for b in bodies]
        recs, wall = asyncio.run(run_level(host, port, path, bodies, conc))
        for r, m in zip(recs, meta):  # tokenize after the level, off the timing path
            r["max_tokens"] = m["max_tokens"]
            if r["ok"]:
                r["retok_tokens"] = len(tok.encode(r["text"], add_special_tokens=False))
                server = (r["usage"] or {}).get("completion_tokens")
                r["out_tokens"] = server if server is not None else r["retok_tokens"]
        s = summarize(recs, meta, wall, conc)
        all_mismatch += s["length_mismatches"]
        print(f"{conc:>4} {s['completed']:>3}/{s['num_requests']:<2} {wall:>7.2f} {s['req_per_s']:>6.2f} "
              f"{s['output_tok_per_s']:>9.1f} {trio(s['ttft_ms']):>22} {trio(s['itl_ms'], 2):>20} "
              f"{trio(s['tpot_ms'], 2, keys=('p50',)):>8} {trio(s['e2e_ms'], 2, 1e-3, ('p50', 'p99')):>14} "
              f"{s['prompt_tokens_mean']:>6.0f} {s['output_tokens_mean'] or 0:>7.1f} {s['length_mismatches']:>7}")
        if s["failed"]:
            print(f"     failures: {s['errors']}")
        results.append(s | {"requests": [
            {k: r.get(k) for k in ("ok", "error", "max_tokens", "out_tokens", "retok_tokens",
                                   "finish_reason", "usage", "e2e")}
            | {"ttft": r["token_times"][0] if r.get("token_times") else None,
               "chunks": len(r.get("token_times") or []), "prompt_tokens": m["prompt_tokens"]}
            for r, m in zip(recs, meta)]})

    usage_ok = all(s["usage_reported"] == s["completed"] for s in results)
    print(f"output token counts from: {'server usage' if usage_ok else 'server usage where present, else re-tokenized text'}")
    if all_mismatch:
        print(f"WARNING: {all_mismatch} request(s) did not generate exactly max_tokens tokens "
              "(engine may ignore `ignore_eos`; try --suppress-eos). See finish_reasons in --out.")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "levels": results}, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
