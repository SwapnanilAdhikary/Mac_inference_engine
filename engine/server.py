"""OpenAI-compatible HTTP server (stdlib asyncio, SSE streaming). One engine thread drives step().

    uv run python -m engine.server --model Qwen/Qwen3-1.7B --port 8000
"""
import argparse, asyncio, json, queue, threading, time, traceback
from .scheduler import SamplingParams

# ponytail: hand-rolled HTTP/1.1 (one request per connection, SSE via Connection: close).
# Swap for uvicorn if keep-alive or HTTP/2 ever matters.


class EngineThread:
    def __init__(self, make_engine):
        """make_engine runs on the engine thread: MLX streams are thread-local, so every array the
        engine touches must be created there."""
        self.cmds, ready = queue.Queue(), threading.Event()

        def boot():
            self.engine = make_engine()
            ready.set()
            self._loop()
        threading.Thread(target=boot, daemon=True).start()
        ready.wait()

    def submit(self, prompt, params, on_token):
        self.cmds.put(("add", prompt, params, on_token))

    def abort(self, holder):
        self.cmds.put(("abort", holder))

    def _loop(self):
        eng, live = self.engine, {}
        while True:
            block = not eng.sched.has_work()
            while True:
                try:
                    cmd = self.cmds.get(block=block)
                except queue.Empty:
                    break
                block = False
                if cmd[0] == "add":
                    _, prompt, params, cb = cmd
                    r = eng.add(prompt, params)
                    live[r.id] = cb
                    cb.request = r
                elif (r := getattr(cmd[1], "request", None)) and not r.done:
                    eng.sched.finish(r, "abort")
                    live.pop(r.id, None)
            try:
                for r, t in eng.step():
                    cb = live.get(r.id)
                    if cb:
                        cb(t, r.finish_reason if r.done else None)
                    if r.done:
                        live.pop(r.id, None)
            except Exception:
                traceback.print_exc()
                for rid, cb in list(live.items()):
                    eng.sched.finish(cb.request, "error")
                    cb(None, "error")
                live.clear()


class Server:
    def __init__(self, engine_thread, tokenizer, model_name, max_len):
        self.et, self.tok, self.name, self.max_len = engine_thread, tokenizer, model_name, max_len

    async def handle(self, reader, writer):
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            line, *hdrs = head.decode("latin-1").split("\r\n")
            method, path, _ = line.split(" ", 2)
            h = {k.strip().lower(): v.strip() for k, v in (x.split(":", 1) for x in hdrs if ":" in x)}
            body = await reader.readexactly(int(h.get("content-length", 0)))
            path = path.split("?")[0]
            if method == "GET" and path in ("/v1/models", "/models"):
                return self.json(writer, 200, {"object": "list", "data": [{"id": self.name, "object": "model"}]})
            if method == "GET" and path == "/health":
                e = self.et.engine
                return self.json(writer, 200, {"free_blocks": e.blocks.num_free(), "num_blocks": e.blocks.num_blocks,
                                               "running": len(e.sched.running), "waiting": len(e.sched.waiting),
                                               "prefix_hit_tokens": e.blocks.hit_tokens,
                                               "prefix_query_tokens": e.blocks.query_tokens,
                                               "spec_drafted": getattr(e, "drafted", 0),
                                               "spec_accepted": getattr(e, "accepted", 0)})
            if method == "POST" and path in ("/v1/completions", "/v1/chat/completions"):
                return await self.generate(json.loads(body or b"{}"), path.endswith("chat/completions"), writer)
            return self.json(writer, 404, {"error": {"message": f"no route {method} {path}"}})
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        except Exception as e:
            traceback.print_exc()
            try:
                self.json(writer, 400, {"error": {"message": str(e), "type": "invalid_request_error"}})
            except Exception:
                pass
        finally:
            try:
                await writer.drain()
                writer.close()
            except Exception:
                pass

    def json(self, w, status, obj):
        data = json.dumps(obj).encode()
        w.write(f"HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {len(data)}\r\n"
                f"Connection: close\r\n\r\n".encode() + data)

    async def generate(self, req, chat, w):
        if chat:
            text = self.tok.apply_chat_template(req["messages"], add_generation_prompt=True, tokenize=False,
                                                **req.get("chat_template_kwargs", {}))
            prompt = self.tok.encode(text, add_special_tokens=False)
        else:
            p = req["prompt"]
            p = p[0] if isinstance(p, list) and p and not isinstance(p[0], int) else p
            prompt = p if isinstance(p, list) else self.tok.encode(p)
        params = SamplingParams(max_tokens=int(req.get("max_tokens") or req.get("max_completion_tokens") or 256),
                                temperature=float(req.get("temperature", 1.0)), top_p=float(req.get("top_p", 1.0)),
                                top_k=int(req.get("top_k", 0) or 0), ignore_eos=bool(req.get("ignore_eos", False)),
                                logit_bias={int(k): float(v) for k, v in (req.get("logit_bias") or {}).items()})
        if len(prompt) + params.max_tokens > self.max_len:
            raise ValueError(f"prompt ({len(prompt)}) + max_tokens ({params.max_tokens}) > {self.max_len}")
        stop = req.get("stop") or []
        stop = [stop] if isinstance(stop, str) else stop
        stream = bool(req.get("stream"))
        usage_in_stream = stream and (req.get("stream_options") or {}).get("include_usage")

        loop, q = asyncio.get_running_loop(), asyncio.Queue()

        def on_token(t, finish):  # called on the engine thread
            loop.call_soon_threadsafe(q.put_nowait, (t, finish))
        self.et.submit(prompt, params, on_token)

        rid, created = f"{'chatcmpl' if chat else 'cmpl'}-{id(q):x}", int(time.time())
        obj = "chat.completion" if chat else "text_completion"

        def chunk(text, finish):
            c = {"index": 0, "finish_reason": finish, "logprobs": None}
            if chat:
                c["delta" if stream else "message"] = {"role": "assistant", "content": text}
            else:
                c["text"] = text
            return {"id": rid, "object": obj + (".chunk" if stream and chat else ""), "created": created,
                    "model": self.name, "choices": [c]}

        if stream:
            w.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                    b"Connection: close\r\n\r\n")
        out, sent, finish = [], "", None
        try:
            while finish is None:
                t, finish = await q.get()
                if t is not None:
                    out.append(t)
                # ponytail: O(n^2) full re-decode per token; switch to windowed detokenize if it shows in profiles
                text = self.tok.decode(out, skip_special_tokens=True)
                if hit := [i for s in stop if (i := text.find(s, max(0, len(sent) - len(s)))) >= 0]:
                    text, finish = text[:min(hit)], "stop"
                    self.et.abort(on_token)
                elif text.endswith("�") and finish is None:
                    continue  # incomplete UTF-8 sequence: wait for more tokens
                delta, sent = text[len(sent):], text
                if stream and (delta or finish):
                    w.write(b"data: " + json.dumps(chunk(delta, finish)).encode() + b"\n\n")
                    await w.drain()
        except (ConnectionError, asyncio.CancelledError):
            self.et.abort(on_token)
            raise
        usage = {"prompt_tokens": len(prompt), "completion_tokens": len(out),
                 "total_tokens": len(prompt) + len(out)}
        if stream:
            if usage_in_stream:
                w.write(b"data: " + json.dumps({"id": rid, "object": obj, "created": created, "model": self.name,
                                                "choices": [], "usage": usage}).encode() + b"\n\n")
            w.write(b"data: [DONE]\n\n")
        else:
            self.json(w, 200, {**chunk(sent, finish), "usage": usage})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--kv-gb", type=float, default=None, help="default: what the GPU has left after weights")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--max-batched-tokens", type=int, default=512)
    ap.add_argument("--max-seqs", type=int, default=64)
    ap.add_argument("--no-prefix-caching", action="store_true")
    ap.add_argument("--quant", type=int, choices=[4], default=None, help="4-bit weights (our kernels)")
    ap.add_argument("--draft", default=None, help="draft model for speculative decoding, e.g. Qwen/Qwen3-0.6B")
    ap.add_argument("--spec-k", type=int, default=4)
    ap.add_argument("--backend", choices=["metal", "mlx", "cuda", "triton", "torch"], default="metal",
                    help="mlx = copy-KV baseline on Mac; cuda/triton = our paged kernels on NVIDIA, torch = copy-KV")
    a = ap.parse_args()
    if a.quant and a.backend != "metal":
        raise SystemExit("--quant needs --backend metal")
    if a.draft and a.backend != "metal":
        raise SystemExit("--draft needs --backend metal")
    from transformers import AutoTokenizer
    from .engine import Engine
    if a.backend in ("cuda", "triton", "torch"):
        from .backend_cuda import TorchBackend
        backend = lambda: TorchBackend(attn=a.backend)
    else:
        from .backend import MLXBackend, MetalBackend
        backend = {"metal": MetalBackend, "mlx": MLXBackend}[a.backend]
    et = EngineThread(lambda: Engine.from_pretrained(a.model, backend(), kv_gb=a.kv_gb, block_size=a.block_size,
                                                     max_batched_tokens=a.max_batched_tokens, max_seqs=a.max_seqs,
                                                     prefix_caching=not a.no_prefix_caching, quant=a.quant,
                                                     draft=a.draft, spec_k=a.spec_k))
    eng = et.engine
    print("warming up (JIT-compiles every kernel once)", flush=True)
    warm = threading.Event()
    et.submit(list(range(1000, 1700)), SamplingParams(max_tokens=8, ignore_eos=True),
              lambda t, finish: finish and warm.set())
    warm.wait()
    eng.blocks.hit_tokens = eng.blocks.query_tokens = 0
    if hasattr(eng, "drafted"):
        eng.drafted = eng.accepted = 0
    cap = eng.blocks.num_blocks * eng.bs
    max_len = min(cap, eng.model.cfg.get("max_position_embeddings", cap))
    print(f"KV cache: {eng.blocks.num_blocks} blocks x {eng.bs} = {cap} tokens "
          f"({cap * eng.model.kv_bytes_per_token / 2**30:.2f} GiB); max_len {max_len}", flush=True)
    srv = Server(et, AutoTokenizer.from_pretrained(a.model), a.model, max_len)

    async def run():
        s = await asyncio.start_server(srv.handle, a.host, a.port, limit=2**24)
        print(f"listening on http://{a.host}:{a.port}", flush=True)
        await s.serve_forever()
    asyncio.run(run())


if __name__ == "__main__":
    main()
