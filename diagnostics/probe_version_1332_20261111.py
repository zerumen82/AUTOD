# -*- coding: utf-8 -*-
"""Verificación del incremento de versión: los 5 sitios sincronizados."""
import re
import subprocess
import sys

sys.path.insert(0, r"D:\PROJECTS\AUTOAUTO")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

r = subprocess.run(
    [sys.executable, "-m", "py_compile", "run.py", "roop/core.py"],
    cwd=r"D:\PROJECTS\AUTOAUTO", capture_output=True,
)
print("py_compile:", "OK" if r.returncode == 0 else r.stderr.decode())
assert r.returncode == 0

from roop.core import APP_VERSION  # noqa: E402

run_src = open(r"D:\PROJECTS\AUTOAUTO\run.py", encoding="utf-8").read()
core_src = open(r"D:\PROJECTS\AUTOAUTO\roop\core.py", encoding="utf-8").read()
agents = open(r"D:\PROJECTS\AUTOAUTO\AGENTS.md", encoding="utf-8").read()
spec = open(r"D:\PROJECTS\AUTOAUTO\spec\00-PRINCIPIOS.md", encoding="utf-8").read()
mem = open(r"D:\PROJECTS\AUTOAUTO\MEMORY.md", encoding="utf-8").read()

banner = re.search(r"AutoAuto (v5\.\d+ \([\d-]+\))", run_src).group(1)
title = re.search(r'"AutoAuto (v5\.\d+)"', core_src).group(1)
print("APP_VERSION   :", APP_VERSION)
print("banner run.py :", banner)
print("titulo core.py:", title)
assert APP_VERSION == banner == "v5.1333 (2026-11-11)", "desincronia banner"
assert title == "v5.1333", "desincronia titulo"
for name, txt in (("AGENTS.md", agents), ("spec/00", spec)):
    assert "v5.1333 (2026-11-11)" in txt, f"desincronia {name}"
assert "v5.1333" in mem, "desincronia MEMORY.md"
print("SINCRONIA OK — run.py, core.py, AGENTS.md, spec/00 y MEMORY.md en", APP_VERSION.split()[0])
