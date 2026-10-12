"""The generator LLM through llama.cpp's llama-server (spec v0.3, section 1.3).

Llama 3.1 8B Instruct, Q4_K_M, on the development desktop (user decision, Oct 10, 2026), with 31
of its 33 layers on the GPU: the fastest setting measured on the Quadro P2200
(RUNLOG_2026-10-08_hotpotqa.md). Used for paraphrases (step 1) and, from stage 3, for answers
and A2 rewrites.

Server is a context manager: it starts llama-server, waits until the model is loaded, and stops
it on exit, so the GPU is free again for the models that run next. Requests go to the
OpenAI-style /v1/chat/completions endpoint, which applies the model's own chat template.
"""

import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

SERVER = Path(r"D:\Han\rex_rag\tools\llama.cpp\llama-server.exe")    # release b11505
MODEL = Path(r"D:\Han\rex_rag\data\models\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
GPU_LAYERS = 31
CTX = 4096
PORT = 8089


class Server:
    def __init__(self, model=MODEL, gpu_layers=GPU_LAYERS, ctx=CTX, port=PORT, log=None):
        self.args = [str(SERVER), "-m", str(model), "-ngl", str(gpu_layers), "-c", str(ctx),
                     "-fa", "on", "-np", "1", "--host", "127.0.0.1", "--port", str(port)]
        self.url = f"http://127.0.0.1:{port}"
        self.log = log
        self.proc = None

    def __enter__(self):
        out = open(self.log, "w", encoding="utf8") if self.log else subprocess.DEVNULL
        self.proc = subprocess.Popen(self.args, stdout=out, stderr=subprocess.STDOUT)
        deadline = time.time() + 600
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.proc.returncode}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                    if r.status == 200:
                        return self
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(1)
        self.__exit__()
        raise TimeoutError("llama-server did not become ready within 10 minutes")

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=60)

    def chat(self, messages, temperature, seed, max_tokens):
        """One completion. Returns (text, info) with token counts and wall-clock seconds.

        The server's prompt cache is off, so every call processes its whole prompt and its
        latency does not depend on what was asked before (section 3.4 latency cost).
        """
        body = json.dumps({"messages": messages, "temperature": temperature, "seed": seed,
                           "max_tokens": max_tokens, "cache_prompt": False}).encode("utf8")
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t = time.time()
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.loads(r.read().decode("utf8"))
        usage = out.get("usage", {})
        return out["choices"][0]["message"]["content"], {
            "seconds": time.time() - t,
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "prompt_tps": out.get("timings", {}).get("prompt_per_second"),
        }
