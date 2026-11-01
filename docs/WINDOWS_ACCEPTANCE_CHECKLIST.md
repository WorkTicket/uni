# Windows Desktop Acceptance Checklist — Universe Engine

**Status by default: NOT TESTED until executed on a real Windows machine.**  
Linux/macOS agent must never mark these as PASSED.  
**Audited SHA:** `7f4f09b86a44bafd5f92bfe84ec04f2385e7ff1d`  
**Host OS (this audit machine):** Windows 11 10.0.26100-SP0, Python 3.14.5, Node v24.15.0.

| # | Test | Windows version / GPU | Command or UI action | Evidence (screenshot/log) | Outcome |
|---|---|---|---|---|---|
| 1 | Fresh clone/install | | `git clone`, `pip install -e .` | | NOT_TESTED |
| 2 | Build `.exe` | | PyInstaller / `UniverseEngine.spec` | | NOT_TESTED |
| 3 | Launch backend and UI | | `universe-observatory --port 8000`, open `http://127.0.0.1:8000/` | | NOT_TESTED |
| 4 | WebGL2 rendering | | 3D viewer at `/` | | NOT_TESTED |
| 5 | Shader compile failures absent | | browser console, `/lab` | | NOT_TESTED |
| 6 | Camera orbit/follow/free/surface | | UI controls | | NOT_TESTED |
| 7 | Galaxy → star → planet search | | catalog navigation | | NOT_TESTED |
| 8 | Select habitat → organism | | `/api/habitat`, `/api/organism` via UI | | NOT_TESTED |
| 9 | Mind Inspector data visible | | organism inspect pane | | NOT_TESTED |
| 10 | Quantum/CHSH laboratory | | research lab | | NOT_TESTED |
| 11 | Brane/multiverse laboratory labeled | | `classification: speculative_research` | | NOT_TESTED |
| 12 | Slow playback and horizon behavior | | timeline scrub | | NOT_TESTED |
| 13 | Save → close → reload → resume | | `/api/save`, `/api/load` | | NOT_TESTED |
| 14 | No hidden server process after exit | | task manager / port check | | NOT_TESTED |
| 15 | Clean Windows user profile | | fresh profile | | NOT_TESTED |

The person who performs checks should record Windows build, GPU, driver, app commit SHA, startup command, exact logs, and screenshot filename.
