"""Local web server for the neuron explorer.  Start with run.bat, then open http://127.0.0.1:8765"""
from __future__ import annotations

import base64
import io
import re
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import backend

app = FastAPI(title="Neuron Explorer")
ex: backend.Explorer | None = None
STATIC = Path(__file__).parent / "static"
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


@app.on_event("startup")
def _startup():
    global ex
    print(f"Loading {backend.MODEL_ID} ... (first load takes a minute or two)")
    ex = backend.Explorer()
    print("Ready on", ex.device_desc)


def get_run(rid):
    if rid not in ex.runs:
        raise HTTPException(404, "That run is no longer in memory - run the prompt again.")
    return ex.runs[rid]


def f32(t):
    return Response(np.ascontiguousarray(t.numpy(), dtype="<f4").tobytes(), media_type="application/octet-stream")


class RunReq(BaseModel):
    prompt: str
    system: str = ""
    max_new_tokens: int = Field(48, ge=0, le=600)
    thinking: bool = False
    use_edits: bool = True
    continue_from: str = ""  # id of the run whose conversation this prompt continues
    role: str = "user"  # "tool" sends the prompt as the result of the tool the model just called
    tools: bool = False  # offer the pretend tools (new chats only; a continued chat keeps its setting)
    image: str = ""  # data: URL of a picture to attach to this message


class Edit(BaseModel):
    kind: str  # neuron | head | steer | group
    layer: int = 0
    index: int = 0
    value: float = 0.0
    vector: str = ""
    enabled: bool = True
    name: str = ""  # group only
    members: list[tuple[int, int]] = []  # group only: (layer, neuron) pairs


class GenReq(BaseModel):
    prompt: str
    system: str = ""
    max_new_tokens: int = Field(48, ge=1, le=600)
    thinking: bool = False
    use_edits: bool = True
    edits: list[Edit] | None = None  # try these instead of the saved edit list
    continue_from: str = ""
    tools: bool = False


class ContrastReq(BaseModel):
    name: str
    prompts: list[str]
    system_a: str
    system_b: str
    max_new_tokens: int = Field(40, ge=4, le=200)


class ExportReq(BaseModel):
    name: str


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/info")
def info():
    return {**ex.info(), "edits": ex.edits, "vectors": ex.vector_list(), "presets": ex.preset_list(),
            "suite": ex.suite(), "lookup": ex.lookup_status(), "tools": [t["function"]["name"] for t in backend.TOOLS],
            "runs": [{"id": r["id"], "label": r["label"]} for r in ex.runs.values()]}


def clean_edits(edits):
    out = []
    for e in edits:
        d = e.model_dump()
        if d["kind"] != "group":
            d.pop("name"), d.pop("members")
        else:
            d["members"] = [list(m) for m in d["members"]]
        err = ex.valid_edit(d)
        if err:
            raise HTTPException(400, err)
        out.append(d)
    return out


def read_image(data_url):
    if not data_url:
        return None
    if len(data_url) > 16_000_000:
        raise HTTPException(400, "That picture is too large (limit about 12 MB).")
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[-1])))
        return ex.prep_image(img)
    except Exception:
        raise HTTPException(400, "Could not read that picture.")


@app.post("/api/run")
def run(req: RunReq):
    if not req.prompt.strip():
        raise HTTPException(400, "Prompt is empty.")
    if req.role not in ("user", "tool"):
        raise HTTPException(400, "Bad role.")
    try:
        r = ex.run(req.prompt, req.system, req.max_new_tokens, req.thinking, req.use_edits,
                   continue_from=req.continue_from or None, role=req.role, tools=req.tools, image=read_image(req.image))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return ex.summary(r)


@app.post("/api/generate")
def generate(req: GenReq):
    """Answer text only (no activations) - used for strength sweeps and the test suite."""
    if not req.prompt.strip():
        raise HTTPException(400, "Prompt is empty.")
    edits = clean_edits(req.edits) if req.edits is not None else (None if req.use_edits else [])
    try:
        t = time.time()
        text = ex.generate(req.prompt, req.system, req.max_new_tokens, req.thinking, edits,
                           req.continue_from or None, req.tools)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"answer": text, "seconds": round(time.time() - t, 1)}


@app.get("/api/run/{rid}")
def run_summary(rid: str):
    return ex.summary(get_run(rid))


@app.get("/api/run/{rid}/mlp")
def run_mlp(rid: str, scope: str = "gen", mode: str = "act"):
    try:
        return f32(ex.mlp_map(get_run(rid), scope, mode))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/run/{rid}/neuron")
def run_neuron(rid: str, layer: int, index: int):
    if not (0 <= layer < ex.L and 0 <= index < ex.d_mlp):
        raise HTTPException(400, "No such neuron.")
    return ex.neuron(get_run(rid), layer, index)


@app.get("/api/run/{rid}/lens")
def run_lens(rid: str, tok: int):
    r = get_run(rid)
    if not 0 <= tok < len(r["ids"]):
        raise HTTPException(400, "No such token.")
    return ex.lens(r, tok)


@app.get("/api/run/{rid}/graph")
def run_graph(rid: str, tok: int):
    r = get_run(rid)
    if not 0 <= tok < len(r["ids"]):
        raise HTTPException(400, "No such token.")
    return ex.graph(r, tok)


@app.get("/api/edits")
def get_edits():
    return ex.edits


@app.put("/api/edits")
def put_edits(edits: list[Edit]):
    ex.edits = clean_edits(edits)
    ex.save_edits()
    return ex.edits


def preset_name(name):
    if not NAME_RE.match(name):
        raise HTTPException(400, "Name must be letters, digits, - or _ (max 40).")
    return name


@app.get("/api/presets")
def presets():
    return ex.preset_list()


@app.put("/api/presets/{name}")
def preset_save(name: str):
    ex.preset_save(preset_name(name))
    return ex.preset_list()


@app.post("/api/presets/{name}/load")
def preset_load(name: str):
    if preset_name(name) not in ex.preset_list():
        raise HTTPException(404, "No such preset.")
    return ex.preset_load(name)


@app.delete("/api/presets/{name}")
def preset_delete(name: str):
    ex.preset_delete(preset_name(name))
    return ex.preset_list()


@app.get("/api/suite")
def suite():
    return ex.suite()


@app.put("/api/suite")
def put_suite(prompts: list[str]):
    prompts = [p.strip() for p in prompts if p.strip()][:40]
    ex.save_suite(prompts)
    return prompts


@app.get("/api/lookup")
def lookup_status():
    return ex.lookup_status()


@app.post("/api/lookup/build")
def lookup_build():
    try:
        ex.start_lookup()
    except ValueError as e:
        raise HTTPException(400, str(e))
    return ex.lookup_status()


@app.get("/api/lookup/neuron")
def lookup_neuron(layer: int, index: int):
    if not (0 <= layer < ex.L and 0 <= index < ex.d_mlp):
        raise HTTPException(400, "No such neuron.")
    try:
        return ex.lookup_neuron(layer, index)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/contrast")
def contrast(req: ContrastReq):
    if not NAME_RE.match(req.name):
        raise HTTPException(400, "Name must be letters, digits, - or _ (max 40).")
    prompts = [p.strip() for p in req.prompts if p.strip()]
    if not prompts:
        raise HTTPException(400, "Give at least one prompt.")
    if len(prompts) > 24:
        raise HTTPException(400, "At most 24 prompts per contrast.")
    return ex.contrast(req.name, prompts, req.system_a, req.system_b, req.max_new_tokens)


@app.get("/api/contrast/{name}/mlp")
def contrast_mlp(name: str):
    if name not in ex.contrasts:
        raise HTTPException(404, "The neuron map for this contrast is not in memory - compute it again.")
    return f32(ex.contrasts[name]["mlp"])


@app.get("/api/vectors")
def vectors():
    return ex.vector_list()


@app.post("/api/export")
def export(req: ExportReq):
    if not NAME_RE.match(req.name):
        raise HTTPException(400, "Name must be letters, digits, - or _ (max 40).")
    return ex.export(req.name)


class FreshStatic(StaticFiles):
    """The app's own scripts: always revalidate, so an updated file is never hidden by the browser cache."""

    async def get_response(self, path, scope):
        r = await super().get_response(path, scope)
        r.headers["Cache-Control"] = "no-cache"
        return r


app.mount("/static", FreshStatic(directory=STATIC), name="static")
app.mount("/vendor/three", StaticFiles(directory=Path(__file__).parent / "node_modules" / "three"), name="three")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765)
