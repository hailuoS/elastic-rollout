"""Keep framework dependencies behind integration modules."""

import ast
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "elastic_rollout"


def core_sources():
    return sorted(path for path in CORE.rglob("*.py") if "integrations" not in path.parts)


def test_core_has_no_direct_framework_imports():
    violations = []
    for path in core_sources():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Call) and node.args:
                target = node.func
                name = target.id if isinstance(target, ast.Name) else getattr(target, "attr", "")
                if name in {"__import__", "import_module", "find_spec"}:
                    value = node.args[0]
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        names = [value.value]
            if any(name.split(".")[0] in {"verl", "areal"} for name in names):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not violations, violations


def test_core_import_and_lifecycle_do_not_load_frameworks():
    modules = sorted(
        {
            ".".join(path.relative_to(ROOT).with_suffix("").parts).removesuffix(".__init__")
            for path in core_sources()
        }
    )
    code = """
import importlib
import importlib.abc
import sys
from types import SimpleNamespace as NS

attempts = []
class DenyFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'verl', 'areal'}:
            attempts.append(fullname)
            raise ImportError('framework dependency forbidden in core: ' + fullname)
sys.meta_path.insert(0, DenyFrameworks())
for name in MODULES:
    importlib.import_module(name)

from elastic_rollout.api import DeploymentSpec, ReplicaSpec, ResourceSlot
from elastic_rollout.resource import StaticResourceProvider
from elastic_rollout.runtime import Controller

for elastic in (False, True):
    driver = NS(start=lambda *a: None, alive=lambda *a: True,
                health=lambda *a: True, warmup=lambda *a: True)
    resources = StaticResourceProvider([ResourceSlot('slot', 'node', 'host', 1)])
    controller = Controller(resources, driver)
    shape = (ReplicaSpec('model', accelerator='gpu', weight_sync='nccl')
             if elastic else ReplicaSpec('model'))
    controller.ensure_deployment(DeploymentSpec('d', shape, 1, elastic_scaling=elastic))
    controller.reconcile()
    controller.reconcile()
    status = controller.deployments['d'].status()
    assert status['ready_replicas'] == 1
assert not attempts, attempts
assert not any(name.split('.')[0] in {'verl', 'areal'} for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", "MODULES = " + json.dumps(modules) + "\n" + code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
