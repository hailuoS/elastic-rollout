"""Execute patched native methods without importing CUDA training frameworks."""

import ast
import asyncio
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("tms_enabled", [False, True])
def test_areal_native_tms_transitions_are_idempotent(tmp_path, tms_enabled):
    source = REPO.parent / "AReaL/areal/engine/megatron_engine.py"
    if not source.exists():
        pytest.skip("adjacent pinned AReaL source unavailable")
    git = Path("C:/Program Files/Git/mingw64/bin/git.exe")
    command = str(git) if git.exists() else shutil.which("git")
    if command is None:
        pytest.skip("git apply unavailable")
    target = tmp_path / "areal/engine/megatron_engine.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_text(encoding="utf-8").encode("utf-8"))
    patch = str(REPO / "patches/megatron/areal-tms-state.patch")
    check = subprocess.run([command, "apply", "--check", patch], cwd=tmp_path, capture_output=True)
    if check.returncode == 0:
        subprocess.run([command, "apply", patch], cwd=tmp_path, check=True, capture_output=True)
    else:
        subprocess.run(
            [command, "apply", "--reverse", "--check", patch],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
    tree = ast.parse(target.read_text(encoding="utf-8"))
    actor = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MegatronEngine")
    methods = [
        n for n in actor.body if isinstance(n, ast.FunctionDef) and n.name in ("offload", "onload")
    ]
    saver = NS(pause=Mock(), resume=Mock())
    namespace = dict(
        is_tms_enabled=lambda: tms_enabled,
        torch_memory_saver=saver,
        current_platform=NS(clear_memory=Mock(), synchronize=Mock()),
        dist=NS(barrier=Mock()),
    )
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(target), "exec"), namespace)
    engine = NS(
        is_offload=False,
        mcore_config=NS(disable_grad_buffers_cpu_backup=False),
        get_device_stats=lambda: NS(log=Mock()),
        cpu_group=object(),
        release_memory_occupation=Mock(),
        resume_memory_occupation=Mock(),
    )
    namespace["onload"](engine)
    for _ in range(3):
        namespace["offload"](engine)
        namespace["offload"](engine)
        assert engine.is_offload is tms_enabled
        namespace["onload"](engine)
        namespace["onload"](engine)
        assert not engine.is_offload
    assert saver.pause.call_count == (3 if tms_enabled else 0)
    assert saver.resume.call_count == (3 if tms_enabled else 0)
    # The pre-existing non-TMS release/resume path still handles every call.
    assert engine.release_memory_occupation.call_count == (0 if tms_enabled else 6)
    assert engine.resume_memory_occupation.call_count == (0 if tms_enabled else 7)


def test_areal_cuda_patch_preserves_native_mindspeed_branch(tmp_path):
    source = REPO.parent / "AReaL/areal/engine/megatron_engine.py"
    if not source.exists():
        pytest.skip("adjacent pinned AReaL source unavailable")
    git = Path("C:/Program Files/Git/mingw64/bin/git.exe")
    command = str(git) if git.exists() else shutil.which("git")
    if command is None:
        pytest.skip("git apply unavailable")
    target = tmp_path / "areal/engine/megatron_engine.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_text(encoding="utf-8").encode("utf-8"))
    patch = str(REPO / "patches/megatron/areal-cuda.patch")
    check = subprocess.run([command, "apply", "--check", patch], cwd=tmp_path, capture_output=True)
    if check.returncode == 0:
        original = ast.parse(target.read_text(encoding="utf-8"))
        subprocess.run([command, "apply", patch], cwd=tmp_path, check=True, capture_output=True)
        patched = ast.parse(target.read_text(encoding="utf-8"))
    else:
        patched = ast.parse(target.read_text(encoding="utf-8"))
        subprocess.run(
            [command, "apply", "--reverse", patch], cwd=tmp_path, check=True, capture_output=True
        )
        original = ast.parse(target.read_text(encoding="utf-8"))
    branch = next(
        n
        for n in patched.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "is_torch_npu_available()"
    )
    start = next(
        i
        for i, n in enumerate(original.body)
        if isinstance(n, ast.Import)
        and n.names[0].name == "areal.engine.megatron_utils.ascend_log_patches"
    )
    stop = next(
        i
        for i, n in enumerate(original.body)
        if isinstance(n, ast.Import) and n.names[0].name == "mbridge"
    )
    assert ast.dump(ast.Module(body=branch.body, type_ignores=[])) == ast.dump(
        ast.Module(body=original.body[start:stop], type_ignores=[])
    )
    post = [
        n
        for n in patched.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "is_torch_npu_available()"
    ][1]
    assert ast.unparse(post.body[0]) == "areal.utils.mbridge_compat.apply_post_mbridge()"
    methods = [
        next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_patch_mindspeed"
        )
        for tree in (original, patched)
    ]
    assert ast.unparse(methods[1].body[0]) == "if not is_npu_available:\n    return"
    assert ast.dump(ast.Module(body=methods[1].body[1:], type_ignores=[])) == ast.dump(
        ast.Module(body=methods[0].body, type_ignores=[])
    )


@pytest.mark.parametrize(
    "enabled,use_lora,is_npu",
    [
        (None, False, False),
        (None, False, True),
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, False, True),
    ],
)
def test_areal_bridge_uses_native_transformer_fusion(tmp_path, enabled, use_lora, is_npu):
    source = REPO.parent / "AReaL/areal/models/mcore/registry.py"
    if not source.exists():
        pytest.skip("adjacent pinned AReaL source unavailable")
    git = Path("C:/Program Files/Git/mingw64/bin/git.exe")
    command = str(git) if git.exists() else shutil.which("git")
    if command is None:
        pytest.skip("git apply unavailable")
    target = tmp_path / "areal/models/mcore/registry.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_text(encoding="utf-8").encode("utf-8"))
    cli = tmp_path / "areal/api/cli_args.py"
    cli.parent.mkdir(parents=True)
    cli.write_bytes(
        (REPO.parent / "AReaL/areal/api/cli_args.py").read_text(encoding="utf-8").encode("utf-8")
    )
    patch = str(REPO / "patches/megatron/areal-bridge-gradient-fusion.patch")
    check = subprocess.run([command, "apply", "--check", patch], cwd=tmp_path, capture_output=True)
    if check.returncode == 0:
        subprocess.run([command, "apply", patch], cwd=tmp_path, check=True, capture_output=True)
    else:
        subprocess.run(
            [command, "apply", "--reverse", "--check", patch],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
    tree = ast.parse(target.read_text(encoding="utf-8"))
    override = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "mcore_config.gradient_accumulation_fusion is not None"
    )
    block = next(
        n.body
        for n in ast.walk(tree)
        if hasattr(n, "body") and isinstance(n.body, list) and override in n.body
    )
    start = block.index(override)
    statements = block[start : start + 3]
    provider = NS(gradient_accumulation_fusion=True)
    tf_config = NS(gradient_accumulation_fusion=enabled)
    namespace = dict(
        provider=provider,
        tf_config=tf_config,
        mcore_config=NS(gradient_accumulation_fusion=enabled),
        use_lora=use_lora,
        is_npu_available=is_npu,
    )
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(target), "exec"), namespace)
    expected = (True if enabled is None else enabled) and not (use_lora or is_npu)
    assert provider.gradient_accumulation_fusion is expected
    assert tf_config.gradient_accumulation_fusion is provider.gradient_accumulation_fusion


@pytest.mark.parametrize("cuda_major", [12, 13])
def test_areal_native_tms_runtime_selection(tmp_path, monkeypatch, cuda_major):
    source = REPO.parent / "AReaL/areal/utils/offload.py"
    if not source.exists():
        pytest.skip("adjacent pinned AReaL source unavailable")
    git = Path("C:/Program Files/Git/mingw64/bin/git.exe")
    command = str(git) if git.exists() else shutil.which("git")
    if command is None:
        pytest.skip("git apply unavailable")
    target = tmp_path / "areal/utils/offload.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_text(encoding="utf-8").encode("utf-8"))
    patch = str(REPO / "patches/megatron/areal-tms-runtime.patch")
    check = subprocess.run([command, "apply", "--check", patch], cwd=tmp_path, capture_output=True)
    if check.returncode == 0:
        subprocess.run([command, "apply", patch], cwd=tmp_path, check=True, capture_output=True)
    else:
        subprocess.run(
            [command, "apply", "--reverse", "--check", patch],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
    tree = ast.parse(target.read_text(encoding="utf-8"))
    method = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_tms_env_vars"
    )
    namespace = {"os": os}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(target), "exec"), namespace)
    monkeypatch.setenv("LD_PRELOAD", "/existing.so")
    selected = f"/torch_memory_saver_hook_mode_preload_cu{cuda_major}.abi3.so:/existing.so"

    @contextmanager
    def configure_subprocess():
        before = os.environ["LD_PRELOAD"]
        os.environ["LD_PRELOAD"] = selected
        try:
            yield
        finally:
            os.environ["LD_PRELOAD"] = before

    monkeypatch.setitem(
        sys.modules, "torch_memory_saver", NS(configure_subprocess=configure_subprocess)
    )
    assert namespace["get_tms_env_vars"]() == {
        "LD_PRELOAD": selected,
        "TMS_INIT_ENABLE": "1",
        "TMS_INIT_ENABLE_CPU_BACKUP": "1",
    }
    assert os.environ["LD_PRELOAD"] == "/existing.so"


@pytest.mark.parametrize("failure", [None, "setup", "transfer"])
def test_verl_native_export_restores_offload(tmp_path, failure):
    source = REPO.parent / "verl/verl/workers/engine_workers.py"
    if not source.exists():
        pytest.skip("adjacent pinned verl source unavailable")
    git = Path("C:/Program Files/Git/mingw64/bin/git.exe")
    command = str(git) if git.exists() else shutil.which("git")
    if command is None:
        pytest.skip("git apply unavailable")
    target = tmp_path / "verl/workers/engine_workers.py"
    target.parent.mkdir(parents=True)
    target.write_bytes(source.read_text(encoding="utf-8").encode("utf-8"))
    patch = str(REPO / "patches/megatron/verl-export-offload.patch")
    check = subprocess.run([command, "apply", "--check", patch], cwd=tmp_path, capture_output=True)
    if check.returncode == 0:
        subprocess.run([command, "apply", patch], cwd=tmp_path, check=True, capture_output=True)
    else:
        # Device validation uses an already-patched independent checkout.
        subprocess.run(
            [command, "apply", "--reverse", "--check", patch],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
    tree = ast.parse(target.read_text(encoding="utf-8"))
    actor = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ActorRolloutRefWorker"
    )
    method = next(
        n for n in actor.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "update_weights"
    )
    method.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(target), "exec"), namespace)
    closed = []

    def export():
        try:
            yield "a", object()
            yield "b", object()
        finally:
            closed.append(True)

    engine = NS(is_param_offload_enabled=True, to=Mock())
    engine.get_per_tensor_param = Mock(
        side_effect=RuntimeError("setup failed") if failure == "setup" else None,
        return_value=(export(), None),
    )

    async def send(weights, global_steps):
        # Setup runs inside the iterator, where the real sender coordinates
        # failures. Its finally closes a partially consumed export on failure.
        engine.get_per_tensor_param.assert_not_called()
        try:
            if failure == "transfer":
                next(weights)
                raise RuntimeError("transfer failed")
            assert len(list(weights)) == 2
            return {"version": global_steps}
        finally:
            weights.close()

    worker = NS(actor=NS(engine=engine), checkpoint_engine=NS(send_weights=send))
    call = namespace["update_weights"](worker, global_steps=1, mode="elastic_nccl")
    if failure:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            asyncio.run(call)
    else:
        assert asyncio.run(call) == {"version": 1}
    engine.to.assert_called_once_with("cpu", model=True, optimizer=False, grad=False)
    if failure != "setup":
        assert closed == [True]
