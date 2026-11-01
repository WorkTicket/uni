# Decision: R16 observatory playback & camera repair (PASS 1-2)

## Problem
The 3D viewer normalizes the entire cosmic timeline to `s in [0,1]` over `TOTAL_SECONDS = 96` wall-clock seconds at 1x, then pauses at `s >= 1`. This makes the universe reach ~100 billion years in ~96 seconds of playback and presents the end of the numerical table (`a=80`) as the physical end of the universe.

## Design

### Authoritative cosmic simulation clock (`src/universe_engine/observatory/static/js/cosmo.js`)
- Holds the current scientifically meaningful cosmic time `t` (seconds), scale factor `a`, redshift `z = 1/a - 1`, background temperature `T = T0/a`, and the validated horizon `a_max`.
- The background table is built on a log-spaced `a` grid from `a0` to `a_max`. `a_max` is a model-validity bound, not the end of physical reality.
- Provides `extendHorizon(new_a_max)` which recomputes the table to a later finite horizon and preserves the current `(t, a)` state. Returns a status describing the new horizon.
- Time and scale factor are mutually consistent: `t = t(a)`, `a = aOfT(t)` via the table; extrapolation beyond the table is NOT allowed (silent extrapolation would be a false claim).
- Redshift is always `z = 1/a - 1` (negative in the future for `a > 1`), never clamped to zero.
- Temperature is `T = T0 / a`, displayed with adaptive precision / scientific notation, never "0.0 K" for nonzero values.
- Labels the region size quantity precisely (e.g. "comoving side length of the periodic representative region"), not a generic physical radius.

### Presentation playback controller (`src/universe_engine/observatory/static/js/app.js`)
- Separate from the physics clock. Controls how fast the viewer advances through *simulated physical time*.
- State: `playing`, `rate` (simulated seconds per wall-clock second), `preset`.
- Presets:
  - Paused (rate 0)
  - Observe (slow, inspectable)
  - Explore (moderately accelerated)
  - Fast Forward (explicitly accelerated)
  - Custom (user-defined valid rate)
  - Cinematic Tour (compressed historical overview, clearly labeled)
- Default: paused at the selected initial epoch (today `a=1` or a supported early epoch), with a clear Start control.
- Changing rate never resets the universe; cosmic time is maintained during camera navigation.
- When the current table reaches its last value (`a >= a_max`), playback pauses and reports "Reached current model horizon — extend to continue" (never "Universe ended").
- An extend-horizon operation is offered; after extending, playback can resume.
- The normalized `[0,1]` timeline is removed as a global law. The scrubber maps to physical time within the current validated horizon.

### Input bindings (PASS 2 core)
- Left drag: orbit target. Right drag: pan. Wheel: smooth scale-aware zoom. Double-click: select/focus. WASD: free-flight/surface movement. Shift: acceleration. F: focus selection. Esc: cancel autopilot/pointer lock. Home: restore overview. Back: return to previous target/scale.

## Tests to add
- Default experience does not finish 100 billion years in an unrequested short cinematic interval.
- Playback speed changes honored without resetting the universe.
- Navigation never changes cosmic time.
- Cosmological time and scale factor agree.
- Future redshift and temperature remain mathematically consistent (z negative for a > 1).
- Configurable horizon can be extended and resumed.
- Model validity boundary produces an informative state (not "Universe ended").

## Scientific status
- FLRW background: implemented, cross-checked vs independent code.
- Future horizon extension: implemented (finite, configurable).
- Negative future redshift: correct per `z = 1/a - 1`.
