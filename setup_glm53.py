"""setup_glm53.py - the GLM-5.3-Flash install (the strata-glm5 engine), run by setup.py for --family glm53.

The Qwen flow in setup.py (families, shards, packs, the prebuilt engine, calibration, images) does not apply to
this model, so its setup is its own, shorter flow on setup.py's helpers: the PC check (AMD cards, RAM, disk), the
questions (cards, context, requests at once), the two GGUF downloads, the engine build (the strata-glm5 target,
HIP only), the tokenizer pack, the run config with the expert budgets sized from the cards' VRAM, and the start.

What the config's numbers come from (README.md, docs/GLM5_NEXT.md): on a Radeon Pro W7900 (48 GB) the engine's
own buffers leave the experts 31.0 / 29.5 GiB (card 0 / card 1 of a two-card split) with one 256K slot, and every
further slot's cache takes 1.2 / 1.8 GiB per 256K of context.  One card holds everything itself: 17.9 GiB of experts
at 256K.  The budgets below are those anchors moved by the card's VRAM and the chosen context and slots.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import setup as S

ROOT = S.ROOT
TAG = "glm53"
MODEL_NAME = "GLM-5.3-Flash"
EXE = "strata-glm5"
CONTEXTS = [32768, 65536, 131072, 262144]
MIN_VRAM_GB = 40.0            # a card of the 48 GB class: the dense weights, the caches and the prefetch ring alone take ~15-25 GB
MIN_BUDGET_GIB = 8.0          # less resident experts than this: the ring and the MTP layer no longer fit
EXPERTS_GIB = 108.0           # every expert of the model (the resident ones + the pinned host complement)
PIN_DEFAULT_GIB = 64.0        # the engine's --pin-gib default: a larger complement must be pinned explicitly
FILES = {
    # name -> (Hugging Face repository, size in bytes)
    "GLM-5.3-Flash-GSQ-RCO-3.0bit.gguf": ("pfeifferj/GLM-5.3-Flash-GSQ-RCO-GGUF", 117481680096),
    "GLM-5.3-Flash-MTP-Q4_K.gguf": ("neuralll/GLM-5.3-Flash-MTP-GGUF", 4588592064),
}
MODEL_FILE, MTP_FILE = list(FILES)
ENV = {"STRATA_GLM5_PF_BUFS": "2", "STRATA_GLM5_LOD_ROUTE": "pool", "STRATA_GLM5_LOD_TOP_POOLS": "1024",
       "STRATA_GLM5_LOD_NOSUM": "1"}


def is_glm_config(path: Path) -> bool:
    """A run config of this model (setup.py's plain start hands those over here)."""
    try:
        cfg = json.loads(path.read_text(encoding="utf-8-sig"))
        return isinstance(cfg, dict) and str(cfg.get("model_name", "")).startswith(MODEL_NAME)
    except (OSError, ValueError):
        return False


def hf_file(name: str) -> str:
    repo, _ = FILES[name]
    return f"{S.hf_endpoint()}/{repo}/resolve/main/{name}"


# ------------------------------------------------------------------------------------------------ the budgets
def budgets(cards: list[dict], ctx: int, parallel: int) -> list[float]:
    """The expert budget per card (GiB): what the VRAM leaves after the engine's own buffers for this context and
    these slots, with a margin.  Two cards: the layers are split between them; one: it holds everything."""
    sets = 1 if parallel <= 1 else parallel + 1          # the batch slots and the solo slot each have a cache
    f = ctx / 262144.0
    if len(cards) == 1:
        v = cards[0]["vram_gb"] * 1e9 / 2 ** 30 if cards[0]["vram_gb"] < 100 else cards[0]["vram_gb"]
        return [round(v - 25.0 - 3.0 * sets * f - 2.0, 1)]
    out = []
    for i, c in enumerate(cards[:2]):
        v = c["vram_gb"]
        overhead, cache = (14.6, 1.2) if i == 0 else (15.1, 1.8)
        out.append(round(v - overhead - cache * sets * f - 1.2, 1))
    return out


def pin_gib(budget_sum: float) -> int | None:
    """--pin-gib when the host complement is over the engine's default pin limit (one card), else None."""
    complement = EXPERTS_GIB - budget_sum
    return int(complement + 6) if complement > PIN_DEFAULT_GIB - 4 else None


# ------------------------------------------------------------------------------------------------ the engine
def build_engine(cards: list[dict], llama: Path) -> Path:
    """Compile strata-glm5 for these cards into engine/ (again only when the source changed)."""
    eng = ROOT / "engine"
    eng.mkdir(exist_ok=True)
    stamp = eng / "BUILD-glm5.json"
    meta = json.loads(stamp.read_text()) if stamp.exists() else {}
    src = S.source_hash(S.ENGINE_SOURCES)
    archs = sorted({c["arch"] for c in cards})
    if (eng / EXE).exists() and meta.get("src") == src and set(archs) <= set(meta.get("archs", [])):
        S.ok("engine already built for this PC")
        return eng
    if not (shutil.which("c++") or shutil.which("g++")) or not shutil.which("git"):
        S.fail("a C++ compiler and git are needed to compile the engine",
               "Ubuntu/Debian: sudo apt install build-essential git   Fedora: sudo dnf install gcc-c++ git")
    root, dirs = S.rocm_root(archs)
    libs = [str(Path(d).parent) for d in dirs[1:]]
    bitcode = next((p for p in (root / "lib" / "llvm" / "amdgcn" / "bitcode", root / "amdgcn" / "bitcode") if p.is_dir()),
                   root / "amdgcn" / "bitcode")
    os.environ.update({"HIP_PLATFORM": "amd", "HIP_COMPILER": "clang", "HIP_RUNTIME": "rocclr", "ROCM_PATH": str(root),
                       "HIP_PATH": str(root)})
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(dirs + [os.environ.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    os.environ["PATH"] = os.pathsep.join([str(root / "bin"), str(root / "llvm" / "bin"), os.environ.get("PATH", "")])
    S.say(f"  Compiling the GLM-5.3-Flash engine for your AMD GPU{'s' if len(archs) > 1 else ''} ({', '.join(archs)}; "
          "10-20 minutes the first time) ...")
    S.cmake_build(ROOT, ROOT / "build-hip", EXE,
                  ["-DSTRATA_ENABLE_HIP=ON", "-DSTRATA_ENABLE_CUDA=OFF", "-DSTRATA_BUILD_TESTS=OFF",
                   "-DSTRATA_PREFILL_MMQ=ON", "-DCMAKE_HIP_ARCHITECTURES=" + ";".join(archs),
                   f"-DCMAKE_HIP_COMPILER={root / 'llvm' / 'bin' / 'clang++'}", f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={root}",
                   "-DCMAKE_PREFIX_PATH=" + ";".join([str(root), *libs]),
                   f"-DCMAKE_HIP_FLAGS=--rocm-path={root} --rocm-device-lib-path={bitcode}",
                   f"-DSTRATA_GGML_DIR={llama}"], None, "")
    shutil.copy2(ROOT / "build-hip" / EXE, eng / EXE)
    stamp.write_text(json.dumps({"source": "local-hip", "backend": "hip", "version": S.source_version(), "archs": archs,
                                 "lib_dirs": dirs, "src": src}, indent=1))
    S.ok(f"engine compiled: {eng / EXE}")
    return eng


# ------------------------------------------------------------------------------------------------ the start
def start(cfg_path: Path, port: int | None, open_browser: bool = True) -> int:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    missing = [p for p in [cfg["exe"], *[x for x in cfg["args"] if x.endswith(".gguf")]] if not Path(p).exists()]
    if missing:
        S.fail(f"{cfg_path.name} refers to missing files: {missing[0]}", "run it again with --setup --family glm53 to repair")
    cfg_path.touch()                                     # the most recently used model
    cmd = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
           "--port", str(port or cfg.get("port", 8080))]
    if open_browser and cfg.get("open_browser") is not False:
        cmd.append("--open")
    cards = cfg.get("gpu") or [0]
    S.say()
    S.say("  " + "-" * 100)
    S.say(f"  Starting {MODEL_NAME} on GPU{'s' if len(cards) > 1 else ''} {', '.join(map(str, cards))}: it loads the "
          "experts onto the cards and pins the rest in RAM (2-4 minutes).")
    S.say("  While it does, YOUR PC CAN BE SLOW OR STOP RESPONDING. That is normal: please wait and don't close this")
    S.say("  window - " + ("the browser opens when it is ready." if "--open" in cmd else "the server says when it is ready."))
    S.say("  Later, closing this window stops the model.")
    S.say("  " + "-" * 100)
    if not S.WIN and os.environ.get("STRATA_EXECV"):
        os.execv(cmd[0], cmd)
    return subprocess.call(cmd)


# ------------------------------------------------------------------------------------------------ the setup
def main(a) -> int:
    """setup.py's parsed arguments; the ones this model uses: --gpu / --gpus, --context, --parallel, --port, --host,
    --api-key, --data-dir, --models-dir, --gguf-dir, --yes, --no-start, --check, --update, --no-browser."""
    S.say(f"Strata - {MODEL_NAME} (the glm5-next engine) on AMD GPUs")
    cfg_path = ROOT / f"strata-{TAG}.json"
    if S.WIN:
        S.fail(f"{MODEL_NAME} runs on Linux with ROCm for now (docs/GLM5_NEXT.md)")
    if cfg_path.is_file() and is_glm_config(cfg_path) and not (a.setup or a.family or a.check or a.update):
        return start(cfg_path, a.port, a.browser is not False)   # installed: just start it

    # ---- 1. the PC
    S.step(1, "checking your PC")
    amd = S.amd_gpus()
    usable = [g for g in amd if S.amd_problem(g) is None and g["vram_gb"] >= MIN_VRAM_GB]
    for g in amd:
        why = S.amd_problem(g) or ("" if g["vram_gb"] >= MIN_VRAM_GB else f"too little VRAM for this model ({MIN_VRAM_GB:.0f} GB needed)")
        S.say(f"  GPU {g['index']}: {g['name']} ({g['vram_gb']:.0f} GB, {g['arch']})" + (f" - {why}" if why else ""))
    if not usable:
        S.fail(f"{MODEL_NAME} needs one or two AMD cards with {MIN_VRAM_GB:.0f} GB of VRAM or more (tested: Radeon Pro "
               "W7900 x2); none found", "the NVIDIA build is untested: docs/GLM5_NEXT.md")
    ram = S.ram_gb()
    S.say(f"  RAM: {ram:.0f} GB")
    data, elsewhere = S.data_folder(a.data_dir)
    models_dir = Path(a.models_dir) if a.models_dir else data / "models"
    if a.check:
        return 0
    if a.update and cfg_path.is_file() and is_glm_config(cfg_path):
        # update.sh: the engine rebuilt for the cards of the existing config, nothing asked, nothing started
        cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
        byid = {g["index"]: g for g in usable}
        cards = [byid[i] for i in (cfg.get("gpu") or [0]) if i in byid] or usable[:1]
        S.step(2, "Python packages and build tools")
        S.pip_install(S.requirement_lines(), "Python packages")
        llama = S.get_llama_cpp()
        S.step(3, "the engine")
        build_engine(cards, llama)
        S.ok("updated (nothing started): ./setup.sh starts the model")
        return 0

    # ---- 2. the choices
    S.step(2, "choices")
    if a.gpus:
        if str(a.gpus).strip().lower() == "all":
            cards = sorted(usable, key=lambda x: (-round(x["vram_gb"]), x["index"]))[:2]
        else:
            byid = {g["index"]: g for g in usable}
            try:
                cards = [byid[int(x)] for x in str(a.gpus).split(",") if x.strip()]
            except (ValueError, KeyError):
                S.fail(f"--gpus takes the numbers of usable AMD GPUs as listed above, e.g. --gpus 0,1, not {a.gpus!r}")
    elif a.gpu is not None:
        byid = {g["index"]: g for g in usable}
        if a.gpu not in byid:
            S.fail(f"--gpu {a.gpu}: not a usable GPU (see the list above)")
        cards = [byid[a.gpu]]
    elif len(usable) >= 2:
        two = sorted(usable, key=lambda x: (-round(x["vram_gb"]), x["index"]))[:2]
        pick = S.ask(f"Use both GPUs {two[0]['index']} and {two[1]['index']} together (recommended: 2x faster, "
                     f"more experts on the cards), or one? [2 = both, 1 = GPU {two[0]['index']} only]", ["2", "1"], "2", a.yes)
        cards = two if pick == "2" else two[:1]
    else:
        cards = usable[:1]
    if len(cards) > 2:
        S.fail("this engine splits the model across at most two cards (--gpus 0,1)")
    ctx = a.context
    if ctx is None:
        ctx = int(S.ask("Context (tokens the model keeps in mind; 262144 = the model's full 256K)",
                        [str(c) for c in CONTEXTS], "262144", a.yes))
    if ctx not in CONTEXTS:
        S.fail(f"--context takes one of {', '.join(map(str, CONTEXTS))}")
    parallel = a.parallel
    if parallel is None:
        parallel = int(S.ask("Requests at once (1 = the fastest answers; 2 = two chats decode together, each with its "
                             "own context)", ["1", "2"], "1", a.yes))
    if parallel not in (1, 2):
        S.fail("--parallel takes 1 or 2 for this model (its decode shares the memory bandwidth)")
    bud = budgets(cards, ctx, parallel)
    if min(bud) < MIN_BUDGET_GIB:
        S.fail(f"the chosen context and slots leave too little VRAM for the experts ({', '.join(f'{b:.1f}' for b in bud)} GiB "
               f"per card; {MIN_BUDGET_GIB:.0f} needed)", "choose a smaller context, one request at a time, or both cards")
    pin = pin_gib(sum(bud))
    need_ram = (EXPERTS_GIB - sum(bud)) + 30
    if ram < need_ram:
        S.fail(f"{ram:.0f} GB of RAM: the experts that do not fit the cards ({EXPERTS_GIB - sum(bud):.0f} GiB) are pinned "
               f"in RAM, and the system needs ~30 GB beside them ({need_ram:.0f} GB in all)",
               "two 48 GB cards need 128 GB of RAM for this model, one card 128 GB as well")
    S.ok(f"GPU{'s' if len(cards) > 1 else ''}: " + " + ".join(f"{c['name']} ({c['vram_gb']:.0f} GB)" for c in cards))
    S.ok(f"context {ctx}, {parallel} request{'s' if parallel > 1 else ''} at once, expert budget "
         + ", ".join(f"{b:.1f} GiB" for b in bud) + (f", {pin} GiB pinned in RAM" if pin else ""))

    # ---- 3. tools
    S.step(3, "Python packages and build tools")
    S.pip_install(S.requirement_lines(), "Python packages")
    llama = S.get_llama_cpp()

    # ---- 4. the model files
    S.step(4, "the model files")
    paths = {}
    for name, (repo, size) in FILES.items():
        dst = None
        if a.gguf_dir and (Path(a.gguf_dir) / name).is_file():
            dst = Path(a.gguf_dir) / name
        else:
            dst = S.find_in([data, *elsewhere], f"models/{name}") or models_dir / name
            if not dst.is_file() or dst.stat().st_size != size:
                free = S.free_gb(dst.parent)
                if free < size / 1e9 + 2 and not dst.is_file():
                    S.fail(f"{free:.0f} GB free on {dst.parent}: {name} is {size / 1e9:.1f} GB", "--data-dir <another drive>")
                S.download(hf_file(name), dst, name)
        if dst.stat().st_size != size:
            S.fail(f"{dst} has {dst.stat().st_size} bytes, expected {size}: a cut-off or different file",
                   "delete it and run setup again (it downloads the whole file)")
        paths[name] = dst
    S.ok("model files present")

    # ---- 5. the engine
    S.step(5, "the engine")
    eng = build_engine(cards, llama)
    if a.update:
        S.ok("updated (nothing started)")
        return 0

    # ---- 6. the tokenizer
    S.step(6, "preparing the tokenizer")
    pack = S.find_in([data, *elsewhere], f"packs/{TAG}") or data / "packs" / TAG
    if not (pack / "tokenizer" / "vocab.json").exists():
        S.run([sys.executable, str(ROOT / "tools" / "strata_tokenizer.py"), "--gguf", str(paths[MODEL_FILE]), "--out", str(pack)],
              env=dict(os.environ, STRATA_GGUF_PY=str(llama / "gguf-py")))
    S.ok(f"tokenizer: {pack / 'tokenizer'}")

    # ---- 7. the run config
    S.step(7, "the run config")
    port = a.port or 8080
    profile = data / "packs" / TAG / "profile.txt"
    profile.parent.mkdir(parents=True, exist_ok=True)
    if not profile.exists():                            # the expert ranking measured on real chats, as the starting point
        shutil.copyfile(ROOT / "data" / TAG / "expert-profile.txt", profile)
    devices = ",".join(str(i) for i in range(len(cards)))   # as HIP numbers them inside HIP_VISIBLE_DEVICES
    args = ["--serve", "--gguf", str(paths[MODEL_FILE]), "--mtp", str(paths[MTP_FILE]), "--devices", devices,
            "--expert-budget-gib", ",".join(f"{b:.1f}" for b in bud),
            "--profile", str(profile), "--profile-save", str(profile),
            "--window", "8", "--prefill-window", "4096", "--spec", "4", "--ctx", str(ctx),
            "--attn", "lod", "--lod-top", "128", "--lod-top-decode", "64"]
    if pin:
        args += ["--pin-gib", str(pin)]
    cfg = {"exe": str(eng / EXE), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
           "model_name": MODEL_NAME, "log": str(ROOT / f"strata-{TAG}.log"), "port": port, "backend": "hip",
           "gpu": [c["index"] for c in cards], "gpus_asked": True,
           "env": {"HIP_VISIBLE_DEVICES": ",".join(str(c["index"]) for c in cards), **ENV},
           "engine_silence_s": 600}
    meta = json.loads((eng / "BUILD-glm5.json").read_text())
    if meta.get("lib_dirs"):
        cfg["lib_dirs"] = meta["lib_dirs"]
    cfg["parallel"] = parallel        # always written: the budgets are sized for these slots (an earlier value must not carry over)
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if a.browser is False:
        cfg["open_browser"] = False
    S.write_setup_config(cfg_path, cfg)
    script = S.write_run_script(TAG, cfg_path, port, a.browser is not False)
    S.ok(f"config written: {cfg_path.name} (start it any time with {script.name})")
    S.say()
    S.say("All set.")
    S.say(f"  API (OpenAI):     http://127.0.0.1:{port}/v1   (any API key; model name: anything)")
    S.say(f"  API (Anthropic):  http://127.0.0.1:{port}/v1/messages")
    S.say(f"  Next time:        just run ./setup.sh (or {script.name}) - it starts right away")
    if a.no_start:
        return 0
    return start(cfg_path, port, a.browser is not False)
