"""Model loading, activation capture and interventions for the neuron explorer.

Hook points per decoder layer (Qwen3.5 text stack):
  * MLP neurons  - the input of `mlp.down_proj`, i.e. act(gate_proj(x)) * up_proj(x), 9216 per layer.
  * Mixer heads  - the input of the token mixer's output projection, split into heads
                   (16 x 256 for full-attention layers, 32 x 128 for gated-delta-net layers).
  * Residual     - the output of the decoder layer.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

MODEL_ID = os.environ.get("NE_MODEL", "Qwen/Qwen3.5-4B")
DEVICE = os.environ.get("NE_DEVICE", "auto")  # auto | cpu
PRECISION = os.environ.get("NE_PRECISION", "auto")  # auto | full | 4bit  (GPU only)
MAX_RUNS = 16
MAX_TOKENS = 1500  # longest conversation (prompt + answer) a single run may cover
RUN_TOKEN_BUDGET = 4000  # total tokens kept in memory across stored runs
ROOT = Path(__file__).parent
EXPORT_DIR = ROOT / "exports"
VECTOR_DIR = ROOT / "vectors"
EDITS_FILE = ROOT / "edits.json"
PRESET_DIR = ROOT / "presets"
SUITE_FILE = ROOT / "suite.json"
CORPUS_FILE = ROOT / "corpus.txt"
LOOKUP_NPZ = ROOT / "lookup.npz"
LOOKUP_JSON = ROOT / "lookup.json"
LOOKUP_K = 6  # example passages kept per neuron and direction
LOOKUP_TOKENS = 64  # each corpus line is cut to this many tokens
MAX_GROUP = 4000  # most neurons one group edit may hold
IMAGE_EDGE = 448  # images are shrunk to this many pixels on the long side (about 200 tokens)
DEFAULT_SUITE = [
    "Tell me about your weekend plans.",
    "Explain how a refrigerator works.",
    "What is the capital of France?",
    "Write a two-line poem about rain.",
    "Give me feedback on my idea to start a bakery.",
    "What is 17 times 6?",
]
# Pretend tools offered to the model in tool-call mode. Nothing is ever executed: you type the result yourself.
TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "calculator", "description": "Evaluate an arithmetic expression and return the number.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "e.g. 12 * (3 + 4)"}},
                       "required": ["expression"]}}},
]


class _Stop(Exception):
    """Raised from a hook to end a forward pass early once the layer we need has run."""


class Explorer:
    def __init__(self):
        self.lock = threading.Lock()
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model, self.device_desc = self._load()
        self.model.eval()
        self.text = self.model.model.language_model
        self.layers = list(self.text.layers)
        self.L = len(self.layers)
        self.d_model = self.text.config.hidden_size
        self.d_mlp = self.layers[0].mlp.down_proj.in_features
        self.layer_types = [l.block_type for l in self.layers]
        self.outproj = []  # per layer: (module, n_heads, head_dim)
        for l in self.layers:
            if l.block_type == "full_attention":
                m = l.self_attn.o_proj
                hd = l.self_attn.head_dim
            else:
                m = l.linear_attn.out_proj
                hd = l.linear_attn.head_v_dim
            self.outproj.append((m, m.in_features // hd, hd))
        self.runs: OrderedDict[str, dict] = OrderedDict()
        self.vectors: dict[str, dict] = {}
        self.contrasts: dict[str, dict] = {}
        self.edits: list[dict] = []
        self._n = 0
        self._processor = None
        self.lookup = None
        self.lookup_state = {"status": "none", "done": 0, "total": 0, "error": ""}
        self._load_vectors()
        self._load_edits()
        self._load_lookup()

    # ------------------------------------------------------------------ edits survive restarts
    def save_edits(self):
        EDITS_FILE.write_text(json.dumps(self.edits, indent=1), encoding="utf-8")

    def valid_edit(self, e):
        """None if the edit is usable with this model, otherwise the reason it is not."""
        try:
            kind, layer = e.get("kind"), int(e.get("layer", 0))
            if kind == "group":
                mem = e.get("members") or []
                if not e.get("name") or not 0 < len(mem) <= MAX_GROUP:
                    return f"A group needs a name and 1 to {MAX_GROUP} neurons."
                if any(not (0 <= int(l) < self.L and 0 <= int(i) < self.d_mlp) for l, i in mem):
                    return "Bad neuron in group."
                return None
            if kind not in ("neuron", "head", "steer") or not 0 <= layer < self.L:
                return "Bad edit."
            if kind == "neuron" and not 0 <= int(e.get("index", 0)) < self.d_mlp:
                return "Bad neuron index."
            if kind == "head" and not 0 <= int(e.get("index", 0)) < self.outproj[layer][1]:
                return "Bad head index."
            if kind == "steer" and e.get("vector") not in self.vectors:
                return f"Unknown steering vector '{e.get('vector')}'."
        except (TypeError, ValueError):
            return "Bad edit."
        return None

    def _load_edits(self):
        try:
            edits = json.loads(EDITS_FILE.read_text(encoding="utf-8"))
            self.edits = [e for e in edits if self.valid_edit(e) is None]
        except FileNotFoundError:
            pass
        except Exception as e:
            print("could not read", EDITS_FILE, e)

    # ------------------------------------------------------------------ presets and the test suite
    def preset_list(self):
        return sorted(f.stem for f in PRESET_DIR.glob("*.json")) if PRESET_DIR.exists() else []

    def preset_save(self, name):
        PRESET_DIR.mkdir(exist_ok=True)
        (PRESET_DIR / f"{name}.json").write_text(json.dumps(self.edits, indent=1), encoding="utf-8")

    def preset_load(self, name):
        edits = json.loads((PRESET_DIR / f"{name}.json").read_text(encoding="utf-8"))
        kept = [e for e in edits if self.valid_edit(e) is None]
        self.edits = kept
        self.save_edits()
        return {"edits": kept, "skipped": len(edits) - len(kept)}

    def preset_delete(self, name):
        (PRESET_DIR / f"{name}.json").unlink(missing_ok=True)

    def suite(self):
        try:
            return [str(p) for p in json.loads(SUITE_FILE.read_text(encoding="utf-8"))]
        except Exception:
            return list(DEFAULT_SUITE)

    def save_suite(self, prompts):
        SUITE_FILE.write_text(json.dumps(prompts, indent=1), encoding="utf-8")

    # ------------------------------------------------------------------ loading
    def _load(self):
        # GPU, full precision (bfloat16, ~9.3 GB) when it fits; otherwise GPU with 4-bit weights
        # (~3.3 GB, activations are a close approximation of the full model); otherwise CPU.
        # NE_PRECISION=full|4bit overrides the choice.
        if DEVICE != "cpu" and torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info()
            name = torch.cuda.get_device_name(0)
            want = PRECISION
            if want == "auto":
                want = "full" if free > 10.5 * 2**30 else "4bit"
            if want == "full":
                model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map={"": 0})
                return model, f"GPU ({name}), full precision"
            if free > 4.2 * 2**30:
                from transformers import BitsAndBytesConfig

                q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=torch.bfloat16)
                model = AutoModelForImageTextToText.from_pretrained(
                    MODEL_ID, quantization_config=q, dtype=torch.bfloat16, device_map={"": 0})
                return model, f"GPU ({name}), 4-bit weights"
            print(f"Only {free / 2**30:.1f} GB of GPU memory is free (need ~4.2 GB) - falling back to CPU.")
        model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.float32)
        return model, "CPU, float32"

    @staticmethod
    def dense(mod):
        """Weight matrix of a Linear layer as a normal tensor (dequantised if it is stored in 4 bits)."""
        w = mod.weight
        qs = getattr(w, "quant_state", None)
        if qs is None:
            return w
        import bitsandbytes as bnb

        return bnb.functional.dequantize_4bit(w.data, qs)

    def info(self):
        return {
            "model": MODEL_ID,
            "device": self.device_desc,
            "n_layers": self.L,
            "d_model": self.d_model,
            "d_mlp": self.d_mlp,
            "layer_types": self.layer_types,
            "heads": [n for _, n, _ in self.outproj],
        }

    # ------------------------------------------------------------------ hooks
    def _edit_hooks(self, edits):
        """Register hooks that apply the active interventions. Returns handles."""
        handles = []
        neuron, head, steer = {}, {}, {}
        for e in edits:
            if not e.get("enabled", True):
                continue
            if e["kind"] == "neuron":
                d = neuron.setdefault(e["layer"], {})
                d[e["index"]] = d.get(e["index"], 1.0) * float(e["value"])
            elif e["kind"] == "group":  # many neurons scaled together; stacks with single-neuron edits
                for l, i in e["members"]:
                    d = neuron.setdefault(int(l), {})
                    d[int(i)] = d.get(int(i), 1.0) * float(e["value"])
            elif e["kind"] == "head":
                head.setdefault(e["layer"], {})[e["index"]] = float(e["value"])
            elif e["kind"] == "steer" and e["vector"] in self.vectors:
                v = self.vectors[e["vector"]]["data"][e["layer"]] * float(e["value"])
                steer[e["layer"]] = steer.get(e["layer"], 0) + v

        for li, d in neuron.items():
            idx = torch.tensor(list(d.keys()))
            val = torch.tensor(list(d.values()))

            def pre(mod, args, idx=idx, val=val):
                x = args[0].clone()
                x[..., idx.to(x.device)] *= val.to(x.device, x.dtype)
                return (x,)

            handles.append(self.layers[li].mlp.down_proj.register_forward_pre_hook(pre))

        for li, d in head.items():
            mod, nh, hd = self.outproj[li]
            scale = torch.ones(nh)
            for k, v in d.items():
                scale[k] = v
            scale = scale.repeat_interleave(hd)

            def pre(mod, args, scale=scale):
                x = args[0]
                return (x * scale.to(x.device, x.dtype),)

            handles.append(mod.register_forward_pre_hook(pre))

        for li, v in steer.items():
            def post(mod, args, out, v=v):
                if isinstance(out, tuple):
                    return (out[0] + v.to(out[0].device, out[0].dtype),) + tuple(out[1:])
                return out + v.to(out.device, out.dtype)

            handles.append(self.layers[li].register_forward_hook(post))
        return handles

    def _capture_hooks(self, store):
        handles = []
        for li, layer in enumerate(self.layers):
            def mlp_pre(mod, args, li=li):
                store["mlp"][li] = args[0][0].detach().float().cpu().half()

            def mlp_post(mod, args, out, li=li):
                store["mlp_norm"][li] = out[0].detach().float().norm(dim=-1).cpu()

            def head_pre(mod, args, li=li):
                _, nh, hd = self.outproj[li]
                x = args[0][0].detach()
                W = self.dense(mod).to(x.dtype)
                norms = []
                for h in range(nh):
                    sl = slice(h * hd, (h + 1) * hd)
                    norms.append((x[:, sl] @ W[:, sl].T).float().norm(dim=-1))
                store["heads"][li] = torch.stack(norms, 1).cpu()
                store["mix_in"][li] = x.float().cpu().half()

            def norm2_post(mod, args, out, li=li):
                x = args[0][0].detach().float()
                store["rms"][li] = x.pow(2).mean(-1).sqrt().cpu()
                store["mlp_in"][li] = out[0].detach().float().cpu()

            def embed_pre(mod, args, kwargs):
                h = args[0] if args else kwargs["hidden_states"]
                store["embed"] = h[0].detach().float().cpu()

            handles.append(layer.post_attention_layernorm.register_forward_hook(norm2_post))
            if li == 0:
                handles.append(layer.register_forward_pre_hook(embed_pre, with_kwargs=True))

            def mix_post(mod, args, out, li=li):
                store["mix_norm"][li] = out[0].detach().float().norm(dim=-1).cpu()

            def resid_post(mod, args, out, li=li):
                o = out[0] if isinstance(out, tuple) else out
                store["resid"][li] = o[0].detach().float().cpu()

            handles.append(layer.mlp.down_proj.register_forward_pre_hook(mlp_pre))
            handles.append(layer.mlp.register_forward_hook(mlp_post))
            handles.append(self.outproj[li][0].register_forward_pre_hook(head_pre))
            handles.append(self.outproj[li][0].register_forward_hook(mix_post))
            handles.append(layer.register_forward_hook(resid_post))
        return handles

    # ------------------------------------------------------------------ running
    def _in_device(self):
        return self.text.embed_tokens.weight.device

    @property
    def processor(self):
        """Image + text processor, loaded the first time a picture is used."""
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(MODEL_ID)
        return self._processor

    @staticmethod
    def prep_image(img):
        img = img.convert("RGB")
        img.thumbnail((IMAGE_EDGE, IMAGE_EDGE))
        return img

    def _conversation(self, prompt, system="", continue_from=None, role="user", tools=False, image=None):
        """Messages for a fresh conversation, or (with `continue_from`) that run's conversation plus
        this new message. Returns (messages, tools offered?)."""
        if continue_from:
            base = self.runs.get(continue_from)
            if base is None:
                raise ValueError("The conversation to continue is no longer in memory - start a new chat.")
            msgs = [dict(m) for m in base["_msgs"]]
            tools = base["tools"]
        else:
            if role != "user":
                raise ValueError("A tool result can only be sent as the next turn of a conversation.")
            msgs = [{"role": "system", "content": system}] if system and system.strip() else []
        m = {"role": role, "content": prompt}
        if image is not None and role == "user":
            m["image"] = image
        msgs.append(m)
        return msgs, bool(tools)

    def _inputs(self, msgs, thinking=False, tools=False):
        """Token ids for the conversation, plus the extra model inputs a picture needs."""
        dev = self._in_device()
        kw = {"add_generation_prompt": True, "enable_thinking": thinking}
        if tools:
            kw["tools"] = TOOLS
        if any(m.get("image") is not None for m in msgs):
            tm = [{"role": m["role"], "content": [{"type": "image", "image": m["image"]}, {"type": "text", "text": m["content"]}]}
                  if m.get("image") is not None else {"role": m["role"], "content": m["content"]} for m in msgs]
            out = self.processor.apply_chat_template(tm, tokenize=True, return_dict=True, return_tensors="pt", **kw)
            extra = {k: v.to(dev) for k, v in out.items() if k not in ("input_ids", "attention_mask")}
            extra["pixel_values"] = extra["pixel_values"].to(self.model.dtype)
            return out["input_ids"].to(dev), extra
        text = self.tok.apply_chat_template([{"role": m["role"], "content": m["content"]} for m in msgs], tokenize=False, **kw)
        return self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(dev), {}

    @staticmethod
    def _extra_for(extra, T):
        """Picture inputs adjusted to a sequence of T tokens (the per-token type ids must match its length)."""
        out = dict(extra)
        m = out.get("mm_token_type_ids")
        if m is not None and m.shape[1] != T:
            out["mm_token_type_ids"] = m[:, :T] if m.shape[1] > T else torch.cat([m, m.new_zeros(1, T - m.shape[1])], 1)
        return out

    def _generate(self, ids, extra, max_new_tokens):
        if max_new_tokens <= 0:
            return ids
        return self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                   do_sample=False, pad_token_id=self.tok.eos_token_id, **extra)

    @torch.no_grad()
    def generate(self, prompt, system="", max_new_tokens=48, thinking=False, edits=None, continue_from=None, tools=False):
        """Just the answer text - no activations recorded. `edits` overrides the current edit list."""
        with self.lock:
            msgs, tools = self._conversation(prompt, system, continue_from, tools=tools)
            ids, extra = self._inputs(msgs, thinking, tools)
            if ids.shape[1] + max_new_tokens > MAX_TOKENS:
                raise ValueError(f"This conversation would be longer than {MAX_TOKENS} tokens.")
            handles = self._edit_hooks(self.edits if edits is None else edits)
            try:
                out = self._generate(ids, extra, max_new_tokens)
            finally:
                for h in handles:
                    h.remove()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return self.tok.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=True).strip()

    @torch.no_grad()
    def run(self, prompt, system="", max_new_tokens=48, thinking=False, use_edits=True, label=None,
            continue_from=None, role="user", tools=False, image=None):
        with self.lock:
            t0 = time.time()
            msgs, tools = self._conversation(prompt, system, continue_from, role, tools, image)
            ids, extra = self._inputs(msgs, thinking, tools)
            n_prompt = ids.shape[1]
            if n_prompt + max_new_tokens > MAX_TOKENS:
                raise ValueError(
                    f"This conversation would be {n_prompt + max_new_tokens} tokens; the explorer records every "
                    f"neuron at every token and stops at {MAX_TOKENS}. Start a new conversation or shorten the answer length.")
            edits = [dict(e) for e in self.edits] if use_edits else []
            handles = self._edit_hooks(edits)
            store = {k: [None] * self.L for k in ("mlp", "mlp_norm", "heads", "mix_norm", "resid", "mix_in", "rms", "mlp_in")}
            try:
                out = self._generate(ids, extra, max_new_tokens)
                t_gen = time.time() - t0
                # one clean pass over prompt + answer with capture hooks (edits still active)
                handles += self._capture_hooks(store)
                self.model(input_ids=out, attention_mask=torch.ones_like(out), **self._extra_for(extra, out.shape[1]))
            finally:
                for h in handles:
                    h.remove()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # hand scratch memory back so other apps keep their VRAM
            seq = out[0].tolist()
            self._n += 1
            rid = f"r{self._n}"
            mlp = torch.stack(store["mlp"])  # L,T,N half
            answer = self.tok.decode(seq[n_prompt:], skip_special_tokens=True)
            # the model's private reasoning is not fed back on later turns, only what follows it
            if thinking and "</think>" not in answer:
                said = "(ran out of room while still thinking)"
            else:
                said = answer.split("</think>")[-1].strip()
            turns = sum(1 for m in msgs if m["role"] in ("user", "tool"))
            kind = "tool result: " if role == "tool" else ""
            run = {
                "id": rid,
                "label": label or (f"turn {turns}: " if turns > 1 else "") + kind + prompt[:60],
                "prompt": prompt,
                "system": next((m["content"] for m in msgs if m["role"] == "system"), ""),
                "_msgs": msgs + [{"role": "assistant", "content": said}],
                "tools": tools,
                "thinking": bool(thinking),
                "extra": extra,
                "ids": seq,
                "tokens": [self.tok.decode([i]) for i in seq],
                "n_prompt": n_prompt,
                "answer": answer,
                "mlp": mlp,
                "heads": store["heads"],
                "resid": torch.stack(store["resid"]),  # L,T,D
                "mlp_norm": torch.stack(store["mlp_norm"]),
                "mix_norm": torch.stack(store["mix_norm"]),
                "mix_in": store["mix_in"],  # per layer: T, heads*head_dim
                "rms": torch.stack(store["rms"]),  # L,T  rms of the residual each MLP reads
                "mlp_in": torch.stack(store["mlp_in"]),  # L,T,D  normalised MLP input
                "embed": store["embed"],  # T,D
                "edits": [e for e in edits if e.get("enabled", True)],
                "gen_seconds": round(t_gen, 1),
                "seconds": round(time.time() - t0, 1),
            }
            self.runs[rid] = run
            # each stored token costs ~1.5 MB of RAM, so keep runs within a token budget as well as a count
            cost = lambda r: len(r["ids"]) * (1.4 if "mlp_base" in r else 1)
            while len(self.runs) > 1 and (len(self.runs) > MAX_RUNS or sum(cost(r) for r in self.runs.values()) > RUN_TOKEN_BUDGET):
                self.runs.popitem(last=False)
            return run

    def summary(self, run):
        resid_norm = run["resid"].norm(dim=-1)
        return {
            "id": run["id"],
            "label": run["label"],
            "prompt": run["prompt"],
            "system": run["system"],
            "messages": [{"role": m["role"], "content": m["content"], "image": m.get("image") is not None} for m in run["_msgs"]],
            "tools": run["tools"],
            "thinking": run["thinking"],
            "tokens": run["tokens"],
            "n_prompt": run["n_prompt"],
            "answer": run["answer"],
            "heads": [h.tolist() for h in run["heads"]],
            "resid_norm": resid_norm.tolist(),
            "mlp_norm": run["mlp_norm"].tolist(),
            "mix_norm": run["mix_norm"].tolist(),
            "edits": run["edits"],
            "gen_seconds": run["gen_seconds"],
            "seconds": run["seconds"],
        }

    def _model_pass(self, run, n=None):
        """Re-read a stored run's tokens (the first n of them) through the model under whatever hooks are set."""
        T = len(run["ids"])
        if n is None or run["extra"]:  # a picture's tokens must all be present
            n = T
        ids = torch.tensor([run["ids"][:n]], device=self._in_device())
        self.model(input_ids=ids, attention_mask=torch.ones_like(ids), **self._extra_for(run["extra"], n))

    def _base_mlp(self, run):
        """Neuron activations of the *unedited* model reading exactly the same tokens as this run."""
        if "mlp_base" not in run:
            got = [None] * self.L
            handles = []
            for li, layer in enumerate(self.layers):
                def pre(mod, args, li=li):
                    got[li] = args[0][0].detach().float().cpu().half()

                handles.append(layer.mlp.down_proj.register_forward_pre_hook(pre))
            try:
                self._model_pass(run)
            finally:
                for h in handles:
                    h.remove()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            run["mlp_base"] = torch.stack(got)
        return run["mlp_base"]

    # ------------------------------------------------------------------ views
    def positions(self, run, scope):
        T = len(run["ids"])
        if scope == "gen":
            p = list(range(run["n_prompt"], T))
            return p or [T - 1]
        if scope == "all":
            return list(range(T))
        i = int(scope)
        if not 0 <= i < T:
            raise ValueError("token index out of range")
        return [i]

    def _unembed_dirs(self, ids):
        """Directions in residual space that raise the logit of each id (final norm gain folded in)."""
        W = self.model.lm_head.weight
        gain = 1.0 + self.text.norm.weight.float()
        return W[torch.tensor(ids, device=W.device)].float() * gain.to(W.device)

    @torch.no_grad()
    def mlp_map(self, run, scope="gen", mode="act"):
        """L x d_mlp float32. mode 'act' = mean activation; 'attr' = mean direct push toward the
        token the model actually produced next (activation x output-weight . unembedding)."""
        pos = self.positions(run, scope)
        if mode == "act":
            return run["mlp"][:, pos, :].float().mean(1)
        if mode == "effect":  # edited minus unedited, both reading this run's exact tokens
            if not run["edits"]:
                raise ValueError("This run had no edits applied, so there is no edit effect to show.")
            with self.lock:
                base = self._base_mlp(run)
            return (run["mlp"][:, pos, :].float() - base[:, pos, :].float()).mean(1)
        if mode != "attr":
            raise ValueError("unknown map mode")
        T = len(run["ids"])
        pos = [p for p in pos if p + 1 < T] or [T - 2]
        nxt = [run["ids"][p + 1] for p in pos]
        with self.lock:
            U = self._unembed_dirs(nxt)
            out = torch.empty(self.L, self.d_mlp)
            for li, layer in enumerate(self.layers):
                W = self.dense(layer.mlp.down_proj)
                D = U.to(W.device, W.dtype) @ W  # P, N
                out[li] = (run["mlp"][li, pos, :].float() * D.float().cpu()).mean(0)
        return out

    @torch.no_grad()
    def neuron(self, run, layer, index, k=14):
        acts = run["mlp"][layer, :, index].float().tolist()
        with self.lock:
            col = self.dense(self.layers[layer].mlp.down_proj)[:, index].float()
            W = self.model.lm_head.weight
            gain = 1.0 + self.text.norm.weight.float()
            logits = (W.float() @ (col.to(W.device) * gain.to(W.device))).cpu()
        top = torch.topk(logits, k)
        bot = torch.topk(-logits, k)
        return {
            "layer": layer,
            "index": index,
            "acts": acts,
            "promotes": [[self.tok.decode([i]), round(v, 3)] for v, i in zip(top.values.tolist(), top.indices.tolist())],
            "suppresses": [[self.tok.decode([i]), round(-v, 3)] for v, i in zip(bot.values.tolist(), bot.indices.tolist())],
        }

    @torch.no_grad()
    def lens(self, run, tok, k=4):
        """Logit lens: what the model would say if it stopped after each layer."""
        h = run["resid"][:, tok, :]  # L, D
        with self.lock:
            norm, head = self.text.norm, self.model.lm_head
            x = norm(h.to(norm.weight.device, norm.weight.dtype))
            probs = torch.softmax(head(x.to(head.weight.device)).float(), -1).cpu()
        top = torch.topk(probs, k, dim=-1)
        return [
            [[self.tok.decode([i]), round(p, 4)] for p, i in zip(ps, is_)]
            for ps, is_ in zip(top.values.tolist(), top.indices.tolist())
        ]

    # ------------------------------------------------------------------ pathways
    @torch.no_grad()
    def graph(self, run, t, first=12, branch=4, depth=3, max_nodes=60, words_for=6):
        """Trace the direct pathways behind the token written after position `t`.

        Start at the output word and ask which neurons / mixer heads / the input word pushed its
        logit most (activation x output weights . unembedding).  Then for each chosen neuron ask
        the same question about *its* activation (linearised through its gate and up weights),
        and repeat.  Everything is a direct path through the residual stream at this position;
        information that arrives from earlier words enters through the heads.
        """
        ids = run["ids"]
        T = len(ids)
        dev = self.model.lm_head.weight.device
        if t + 1 < T:
            target = ids[t + 1]
        else:
            norm, head = self.text.norm, self.model.lm_head
            target = int(head(norm(run["resid"][-1, t].to(dev, norm.weight.dtype))).argmax())
        with self.lock:
            acts = run["mlp"][:, t, :].float().to(dev)  # L,N
            mixin = [m[t].float().to(dev) for m in run["mix_in"]]
            emb = run["embed"][t].to(dev)
            final_rms = run["resid"][-1, t].pow(2).mean().sqrt().item()

            nodes, edges = {}, []
            out_id = "out"
            nodes[out_id] = {"id": out_id, "kind": "out", "layer": self.L, "index": 0, "weight": 1.0,
                             "depth": 0, "label": self.tok.decode([target]), "value": 0.0}
            # frontier entries: (node id, direction d in residual space, neuron layers < nl, head layers < hl)
            d0 = self._unembed_dirs([target])[0].to(dev) / final_rms
            frontier = [(out_id, d0, self.L, self.L)]
            cols = {}  # neuron node id -> output weight column, for labelling
            hdir = {}  # head node id -> (|value|, direction it was pushing along), for tracing back to words

            for level in range(depth):
                if not frontier:
                    break
                D = torch.stack([f[1] for f in frontier])  # K, d_model
                K = len(frontier)
                nl = torch.tensor([f[2] for f in frontier], device=dev)
                hl = torch.tensor([f[3] for f in frontier], device=dev)
                cand_val, cand_key = [], []  # per layer tensors (K, n) and key info
                for li in range(int(max(nl.max(), hl.max()))):
                    if (nl > li).any():
                        W = self.dense(self.layers[li].mlp.down_proj)
                        c = (D.to(W.dtype) @ W).float() * acts[li]  # K,N
                        c[nl <= li] = 0
                        v, i = torch.topk(c.abs(), 8, dim=1)
                        cand_val.append(torch.gather(c, 1, i))
                        cand_key.append([[("neuron", li, int(x)) for x in row] for row in i.tolist()])
                        del W
                    if (hl > li).any():
                        mod, nh, hd = self.outproj[li]
                        W = self.dense(mod)
                        c = ((D.to(W.dtype) @ W).float() * mixin[li]).view(K, nh, hd).sum(-1)  # K,H
                        c[hl <= li] = 0
                        cand_val.append(c)
                        cand_key.append([[("head", li, h) for h in range(nh)]] * K)
                        del W
                cand_val.append((D @ emb)[:, None])
                cand_key.append([[("embed", -1, t)]] * K)
                vals = torch.cat(cand_val, 1).cpu()  # K, C
                want = first if level == 0 else branch
                top = torch.topk(vals.abs(), min(want, vals.shape[1]), dim=1).indices.tolist()
                nxt = []
                for k, (pid, _, _, _) in enumerate(frontier):
                    keys = [x for part in cand_key for x in part[k]]
                    best = vals[k, top[k][0]].abs().item() or 1.0
                    for ci in top[k]:
                        kind, li, idx = keys[ci]
                        val = vals[k, ci].item()
                        rel = abs(val) / best
                        if rel < 0.15:
                            continue
                        nid = f"{kind[0]}{li}_{idx}"
                        w = nodes[pid]["weight"] * rel
                        if nid not in nodes:
                            if len(nodes) >= max_nodes:
                                continue
                            nodes[nid] = {"id": nid, "kind": kind, "layer": li, "index": idx, "weight": w,
                                          "depth": level + 1, "label": "", "value": 0.0}
                            if kind == "neuron":
                                nodes[nid]["value"] = acts[li, idx].item()
                                nxt.append((nid, li, idx))
                            elif kind == "embed":
                                nodes[nid]["label"] = run["tokens"][t]
                        else:
                            nodes[nid]["weight"] = max(nodes[nid]["weight"], w)
                        if kind == "head" and abs(val) > hdir.get(nid, (0, None))[0]:
                            hdir[nid] = (abs(val), D[k])
                        edges.append({"src": nid, "dst": pid, "value": val, "w": w})
                # directions for the next level: how each chosen neuron's activation depends on the residual
                frontier = []
                by_layer = {}
                for nid, li, idx in nxt:
                    by_layer.setdefault(li, []).append((nid, idx))
                for li, items in by_layer.items():
                    mlp = self.layers[li].mlp
                    idx = torch.tensor([i for _, i in items], device=dev)
                    Wg = self.dense(mlp.gate_proj)[idx].float()
                    Wu = self.dense(mlp.up_proj)[idx].float()
                    Wd = self.dense(mlp.down_proj)[:, idx].float()
                    x = run["mlp_in"][li, t].to(dev)
                    g, u = Wg @ x, Wu @ x
                    sg = torch.sigmoid(g)
                    dact = (sg * (1 + g * (1 - sg)) * u)[:, None] * Wg + (g * sg)[:, None] * Wu
                    gain = (1.0 + self.layers[li].post_attention_layernorm.weight.float()).to(dev)
                    dres = dact * gain / run["rms"][li, t].item()
                    for k, (nid, _) in enumerate(items):
                        cols[nid] = Wd[:, k]
                        if level + 1 < depth:
                            frontier.append((nid, dres[k], li, li + 1))

            if cols:
                names = list(cols)
                C = torch.stack([cols[n] for n in names])
                gain = 1.0 + self.text.norm.weight.float().to(dev)
                W = self.model.lm_head.weight
                logits = ((C * gain).to(W.dtype) @ W.T).float()
                hi, lo = logits.argmax(1).tolist(), logits.argmin(1).tolist()
                for n, a, b in zip(names, hi, lo):
                    # a neuron with negative activation pushes toward its "suppressed" side
                    nodes[n]["label"] = self.tok.decode([a if nodes[n]["value"] >= 0 else b])

            # follow the strongest heads back to the earlier words they were reading
            note = ""
            top_heads = sorted(hdir, key=lambda n: -nodes[n]["weight"])[:words_for]
            if top_heads and t > 0:
                try:
                    src = self._head_sources(run, t, [(n, nodes[n]["layer"], nodes[n]["index"], hdir[n][1]) for n in top_heads])
                except Exception as e:  # tracing back is a bonus - never lose the rest of the pathway over it
                    src, note = {}, f"{type(e).__name__}: {e}"[:200]
                n_words = 0
                for nid in top_heads:
                    attr = src.get(nid)
                    if attr is None:
                        continue
                    attr[0] = 0  # the very first token is a start marker every position leans on; it is not a "word"
                    best = attr.abs().max().item() or 1.0
                    for p in torch.topk(attr.abs(), min(3, len(attr))).indices.tolist():
                        rel = abs(attr[p].item()) / best
                        if p == t or rel < 0.3 or n_words >= 16:
                            continue  # the current position is already the direct path drawn above
                        wid = f"w{p}"
                        w = nodes[nid]["weight"] * rel
                        if wid not in nodes:
                            nodes[wid] = {"id": wid, "kind": "word", "layer": -1, "index": p, "weight": w,
                                          "depth": nodes[nid]["depth"] + 1, "label": run["tokens"][p], "value": 0.0}
                            n_words += 1
                        else:
                            nodes[wid]["weight"] = max(nodes[wid]["weight"], w)
                        edges.append({"src": wid, "dst": nid, "value": attr[p].item(), "w": w})
        return {"token": t, "target": self.tok.decode([target]), "nodes": list(nodes.values()), "edges": edges, "note": note}

    def _head_sources(self, run, t, items):
        """For mixer heads at position t, how much each earlier position fed what the head wrote.

        `items` is a list of (key, layer, head, direction). For each, the head's push along `direction`
        is differentiated with respect to the layer's input at every position, and gradient x input
        gives one number per position. Works the same for attention and delta-net layers.
        Caller holds the lock."""
        out = {}
        by_layer = {}
        for key, li, h, d in items:
            by_layer.setdefault(li, []).append((key, h, d))
        for li, group in by_layer.items():
            mod, nh, hd = self.outproj[li]
            cap = {}

            def layer_pre(m, args, kwargs):
                h = args[0] if args else kwargs["hidden_states"]
                x = h.detach().clone().requires_grad_(True)
                cap["x"] = x
                if args:
                    return (x,) + tuple(args[1:]), kwargs
                return args, {**kwargs, "hidden_states": x}

            def out_pre(m, args):
                cap["h"] = args[0]

            def out_post(m, args, o):
                raise _Stop()

            handles = self._edit_hooks(run["edits"])
            handles += [self.layers[li].register_forward_pre_hook(layer_pre, with_kwargs=True),
                        mod.register_forward_pre_hook(out_pre), mod.register_forward_hook(out_post)]
            try:
                with torch.enable_grad():
                    try:
                        self._model_pass(run, t + 1)
                    except _Stop:
                        pass
                    W = self.dense(mod).float()
                    for key, h, d in group:
                        sl = slice(h * hd, (h + 1) * hd)
                        s = (cap["h"][0, t, sl].float() * (d.float() @ W[:, sl])).sum()
                        g = torch.autograd.grad(s, cap["x"], retain_graph=True)[0][0]
                        out[key] = (g.float() * cap["x"][0].detach().float()).sum(-1)[: t + 1].cpu()
            finally:
                for hnd in handles:
                    hnd.remove()
                cap.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return out

    # ------------------------------------------------------------------ what does this neuron do? (corpus lookup)
    def corpus(self):
        try:
            lines = CORPUS_FILE.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        return [l.strip().replace("\\n", "\n") for l in lines if l.strip() and not l.startswith("#")]

    def _load_lookup(self):
        try:
            meta = json.loads(LOOKUP_JSON.read_text(encoding="utf-8"))
            z = np.load(LOOKUP_NPZ)
            if z["pos_v"].shape[:2] != (self.L, self.d_mlp) or meta.get("model") != MODEL_ID:
                return
            self.lookup = {"texts": meta["texts"], **{k: z[k] for k in z.files}}
            self.lookup_state = {"status": "ready", "done": len(meta["texts"]), "total": len(meta["texts"]), "error": ""}
        except FileNotFoundError:
            pass
        except Exception as e:
            print("could not read the neuron lookup:", e)

    def lookup_status(self):
        return {**self.lookup_state, "passages": len(self.lookup["texts"]) if self.lookup else 0,
                "corpus": len(self.corpus())}

    def _lookup_ids(self, text):
        bos = self.tok.convert_tokens_to_ids("<|endoftext|>")  # a start marker, so no real word sits at position 0
        return [bos] + self.tok(text, add_special_tokens=False).input_ids[:LOOKUP_TOKENS]

    def start_lookup(self):
        if self.lookup_state["status"] == "building":
            return
        texts = self.corpus()
        if not texts:
            raise ValueError(f"{CORPUS_FILE.name} is missing or empty.")
        self.lookup_state = {"status": "building", "done": 0, "total": len(texts), "error": ""}
        threading.Thread(target=self._build_lookup, args=(texts,), daemon=True).start()

    @torch.no_grad()
    def _build_lookup(self, texts):
        """Read every corpus passage once with the unedited model and remember, for every neuron,
        the passages (and the word in each) where it fired hardest - positively and negatively."""
        try:
            dev, K = self._in_device(), LOOKUP_K
            st = {key: [torch.full((self.L, self.d_mlp, K), float("-inf"), device=dev),
                        torch.zeros((self.L, self.d_mlp, K), dtype=torch.long, device=dev),
                        torch.zeros((self.L, self.d_mlp, K), dtype=torch.long, device=dev)] for key in ("pos", "neg")}
            cur = {"i": 0}
            handles = []
            for li, layer in enumerate(self.layers):
                def pre(mod, args, li=li):
                    x = args[0][0, 1:].float()  # skip the start marker
                    for key, sign in (("pos", 1.0), ("neg", -1.0)):
                        V, S, P = st[key]
                        m, am = (x * sign).max(0)  # one candidate per neuron per passage keeps the examples varied
                        top = torch.cat([V[li], m[:, None]], 1).topk(K, 1)
                        V[li] = top.values
                        S[li] = torch.cat([S[li], torch.full_like(am, cur["i"])[:, None]], 1).gather(1, top.indices)
                        P[li] = torch.cat([P[li], am[:, None] + 1], 1).gather(1, top.indices)

                handles.append(layer.mlp.down_proj.register_forward_pre_hook(pre))
            try:
                for i, text in enumerate(texts):
                    with self.lock:  # one passage at a time, so normal runs can slip in between
                        cur["i"] = i
                        ids = torch.tensor([self._lookup_ids(text)], device=dev)
                        if ids.shape[1] > 1:
                            self.text(input_ids=ids, attention_mask=torch.ones_like(ids))
                    self.lookup_state["done"] = i + 1
            finally:
                for h in handles:
                    h.remove()
            arrays = {}
            for key, (V, S, P) in st.items():
                arrays[key + "_v"] = (V * (1.0 if key == "pos" else -1.0)).cpu().numpy().astype(np.float16)
                arrays[key + "_s"] = S.cpu().numpy().astype(np.int16)
                arrays[key + "_p"] = P.cpu().numpy().astype(np.int16)
            del st
            np.savez_compressed(LOOKUP_NPZ, **arrays)
            LOOKUP_JSON.write_text(json.dumps({"model": MODEL_ID, "device": self.device_desc, "texts": texts}), encoding="utf-8")
            self.lookup = {"texts": texts, **arrays}
            self.lookup_state = {"status": "ready", "done": len(texts), "total": len(texts), "error": ""}
        except Exception as e:
            self.lookup_state = {"status": "error", "done": 0, "total": 0, "error": f"{type(e).__name__}: {e}"[:300]}
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @torch.no_grad()
    def lookup_neuron(self, layer, index):
        """The corpus passages where this neuron fired hardest, with its activation on every word of each."""
        lk = self.lookup
        if lk is None:
            raise ValueError("The neuron lookup has not been built yet.")
        picks = {}
        for key in ("pos", "neg"):
            for k in range(LOOKUP_K):
                v = float(lk[key + "_v"][layer, index, k])
                if np.isfinite(v) and abs(v) > 1e-4:
                    picks.setdefault(key, []).append((v, int(lk[key + "_s"][layer, index, k]), int(lk[key + "_p"][layer, index, k])))
        acts = {}
        mod = self.layers[layer].mlp.down_proj
        got = {}

        def pre(m, args):
            got["a"] = args[0][0, :, index].float().cpu().tolist()
            raise _Stop()

        with self.lock:
            h = mod.register_forward_pre_hook(pre)
            try:
                for si in {s for items in picks.values() for _, s, _ in items}:
                    ids = self._lookup_ids(lk["texts"][si])
                    t = torch.tensor([ids], device=self._in_device())
                    try:
                        self.text(input_ids=t, attention_mask=torch.ones_like(t))
                    except _Stop:
                        pass
                    acts[si] = ([self.tok.decode([i]) for i in ids[1:]], got["a"][1:])
            finally:
                h.remove()
        out = {"layer": layer, "index": index, "passages": len(lk["texts"])}
        for key in ("pos", "neg"):
            out[key] = [{"value": round(v, 3), "peak": p - 1, "tokens": acts[s][0], "acts": [round(a, 3) for a in acts[s][1]]}
                        for v, s, p in picks.get(key, [])]
        return out

    # ------------------------------------------------------------------ contrast / steering
    def contrast(self, name, prompts, system_a, system_b, max_new_tokens=40):
        """Run every prompt under two system prompts with the unedited model and diff them (A - B)."""
        acc = {s: {"mlp": 0, "resid": 0, "heads": [0] * self.L, "n": 0} for s in "ab"}
        samples = []
        resid_norm = 0
        for p in prompts:
            row = {"prompt": p}
            for side, system in (("a", system_a), ("b", system_b)):
                r = self.run(p, system, max_new_tokens, use_edits=False, label=f"[{name}:{side.upper()}] {p[:40]}")
                pos = self.positions(r, "gen")
                a = acc[side]
                a["mlp"] = a["mlp"] + r["mlp"][:, pos, :].float().mean(1)
                a["resid"] = a["resid"] + r["resid"][:, pos, :].mean(1)
                a["heads"] = [x + h[pos].mean(0) for x, h in zip(a["heads"], r["heads"])]
                a["n"] += 1
                resid_norm = resid_norm + r["resid"][:, pos, :].norm(dim=-1).mean(1)
                row[side] = r["answer"]
                row[side + "_id"] = r["id"]
            samples.append(row)
        n = len(prompts)
        mlp_diff = (acc["a"]["mlp"] - acc["b"]["mlp"]) / n
        vec = (acc["a"]["resid"] - acc["b"]["resid"]) / n  # L, D
        heads_diff = [((x - y) / n).tolist() for x, y in zip(acc["a"]["heads"], acc["b"]["heads"])]
        typical = (resid_norm / (2 * n)).tolist()
        self.vectors[name] = {"data": vec, "typical_norm": typical, "system_a": system_a, "system_b": system_b}
        self._save_vector(name)
        self.contrasts[name] = {"mlp": mlp_diff, "heads": heads_diff, "samples": samples}
        return {
            "name": name,
            "samples": samples,
            "heads": heads_diff,
            "vector_norm": vec.norm(dim=-1).tolist(),
            "typical_norm": typical,
        }

    def _save_vector(self, name):
        from safetensors.torch import save_file

        VECTOR_DIR.mkdir(exist_ok=True)
        v = self.vectors[name]
        save_file({"vector": v["data"].contiguous()}, VECTOR_DIR / f"{name}.safetensors")
        meta = {k: v[k] for k in ("typical_norm", "system_a", "system_b")}
        (VECTOR_DIR / f"{name}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    def _load_vectors(self):
        from safetensors.torch import load_file

        if not VECTOR_DIR.exists():
            return
        for f in VECTOR_DIR.glob("*.safetensors"):
            try:
                data = load_file(f)["vector"]
                meta = json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))
                if data.shape == (self.L, self.d_model):
                    self.vectors[f.stem] = {"data": data, **meta}
            except Exception as e:  # a bad file should not stop the app
                print("skipping vector", f, e)

    def vector_list(self):
        return [
            {"name": n, "norm": v["data"].norm(dim=-1).tolist(), "typical_norm": v["typical_norm"],
             "system_a": v.get("system_a", ""), "system_b": v.get("system_b", ""), "has_map": n in self.contrasts}
            for n, v in self.vectors.items()
        ]

    # ------------------------------------------------------------------ export
    @torch.no_grad()
    def export(self, name):
        """Bake neuron/head scalings into a full copy of the model. Steering vectors cannot be
        expressed in the weights (the architecture has no bias terms), so they are written as a
        side file plus a small loader script."""
        from safetensors.torch import save_file

        out = EXPORT_DIR / name
        out.mkdir(parents=True, exist_ok=True)
        active = [e for e in self.edits if e.get("enabled", True)]
        # Always bake into a fresh full-precision copy read from disk, so the saved model does not
        # depend on how the live one is loaded (4-bit on GPU, float32 on CPU, ...).
        full = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
        layers = full.model.language_model.layers
        for e in active:
            if e["kind"] == "group":
                for l, i in e["members"]:
                    W = layers[int(l)].mlp.down_proj.weight
                    W[:, int(i)] = (W[:, int(i)].float() * float(e["value"])).to(W.dtype)
                continue
            if e["kind"] == "neuron":
                W = layers[e["layer"]].mlp.down_proj.weight
                sl = slice(e["index"], e["index"] + 1)
            elif e["kind"] == "head":
                lyr = layers[e["layer"]]
                W = (lyr.self_attn.o_proj if lyr.block_type == "full_attention" else lyr.linear_attn.out_proj).weight
                hd = self.outproj[e["layer"]][2]
                sl = slice(e["index"] * hd, (e["index"] + 1) * hd)
            else:
                continue
            W[:, sl] = (W[:, sl].float() * float(e["value"])).to(W.dtype)
        full.save_pretrained(out)
        self.tok.save_pretrained(out)
        del full, layers
        for f in ("preprocessor_config.json", "video_preprocessor_config.json", "chat_template.jinja"):
            try:
                from huggingface_hub import hf_hub_download
                import shutil

                if not (out / f).exists():
                    shutil.copy(hf_hub_download(MODEL_ID, f), out / f)
            except Exception:
                pass
        steer = [e for e in active if e["kind"] == "steer" and e["vector"] in self.vectors]
        if steer:
            total = torch.zeros(self.L, self.d_model)
            for e in steer:
                total[e["layer"]] += self.vectors[e["vector"]]["data"][e["layer"]] * float(e["value"])
            save_file({"steering": total}, out / "steering.safetensors")
            (out / "load_steered.py").write_text(LOADER, encoding="utf-8")
        (out / "neuron_explorer_edits.json").write_text(json.dumps(active, indent=1), encoding="utf-8")
        return {
            "path": str(out),
            "baked": sum(1 for e in active if e["kind"] in ("neuron", "head", "group")),
            "steering": len(steer),
        }


LOADER = '''"""Load this exported model with its steering vectors applied.

Neuron and head edits are already baked into the weights. Steering vectors are added to the
residual stream at run time by the hooks below.

    from load_steered import load
    model, tok = load()
"""
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText, AutoTokenizer

HERE = Path(__file__).parent


def load(**kwargs):
    model = AutoModelForImageTextToText.from_pretrained(HERE, **kwargs)
    tok = AutoTokenizer.from_pretrained(HERE)
    steering = load_file(HERE / "steering.safetensors")["steering"]
    for i, layer in enumerate(model.model.language_model.layers):
        v = steering[i]
        if not v.abs().sum():
            continue

        def hook(mod, args, out, v=v):
            if isinstance(out, tuple):
                return (out[0] + v.to(out[0].device, out[0].dtype),) + tuple(out[1:])
            return out + v.to(out.device, out.dtype)

        layer.register_forward_hook(hook)
    return model, tok
'''
