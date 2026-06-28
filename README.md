# SpaceX Recovery Dynamics

Booster landing guidance and ASDS recovery operations for Falcon 9 first stage.

## Architecture

**Double Helix (Alpha + Omega)**

- **Alpha** (`src/alpha/landing_guidance.py`): Suicide burn timing, throttle computation, grid fin control, descent guidance.
- **Omega** (`src/omega/recovery_controller.py`): Drone ship positioning, station-keeping, wave compensation, fleet coordination.

## Features

- Suicide burn (hoverslam) calculator
- Throttle percentage computation
- Grid fin deflection control
- Barge station-keeping with position hold
- Wave compensation for landing pad stability
- Fleet management with landing assignment
- Zero external dependencies
