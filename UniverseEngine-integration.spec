# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Universe Engine observatory — built from the verified
# integration branch integration/observatory-reconciliation (SHA 17326c1).
# All paths are absolute so the build is reproducible and does not depend on
# the current working directory or an editable install of another worktree.
import sys
from PyInstaller.utils.hooks import collect_submodules

WORKTREE = r"C:\Users\Slay3r\Downloads\universe-starter-build-worktree"
SRC = WORKTREE + r"\src"

hiddenimports = ["websockets"]
hiddenimports += collect_submodules("uvicorn")
hiddenimports += collect_submodules("universe_engine")

a = Analysis(
    [WORKTREE + r"\packaging\entry.py"],
    pathex=[SRC],
    binaries=[],
    datas=[
        (WORKTREE + r"\src\universe_engine\observatory\static", "universe_engine/observatory/static"),
        (WORKTREE + r"\config\reference.yaml", "config"),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pkg_resources", "setuptools"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="UniverseEngine",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
