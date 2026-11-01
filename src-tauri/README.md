# Universe Engine - Tauri Desktop Application

This is a Tauri 2 desktop application for the Universe Engine scientific simulator.

## Prerequisites

- Rust toolchain (stable)
- Node.js (for frontend, optional)
- Python 3.10+ with the universe-engine package installed

## Build

```bash
# Install Tauri CLI
cargo install tauri-cli

# Build the desktop application
cargo tauri build
```

This produces a Windows MSI installer in `src-tauri/target/release/bundle/msi/`.

## Runtime

The application bundles a Python scientific backend. On first launch, it starts the
FastAPI observatory server on localhost and opens the desktop UI (WebView2 on Windows).

## Notes

- Rust is required to compile Tauri. If Rust is not available, use the Python
  launcher instead: double-click `Launch_Universe.bat` or run
  `python -m universe_engine.launcher`.
- The Tauri frontend reuses the existing observatory static files under
  `src/universe_engine/observatory/static/`.
