"""Booster landing guidance — powered descent, suicide burn, and grid fin control.

Implements the guidance algorithms for Falcon 9 first stage recovery.
Suicide burn timing, attitude control, and landing pad targeting.
Pure math, zero external dependencies.
"""

import math
import time
from dataclasses import dataclass
from typing import Optional

G0 = 9.80665
RHO_SL = 1.225
R_EARTH = 6371000


@dataclass
class BoosterState:
    altitude: float
    velocity: float
    mass: float
    thrust: float
    isp: float
    drag_coeff: float
    cross_section: float
    lat: float = 0.0
    lon: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    @property
    def weight(self) -> float:
        return self.mass * G0

    @property
    def twr(self) -> float:
        if self.weight <= 0:
            return float("inf")
        return self.thrust / self.weight

    @property
    def exhaust_velocity(self) -> float:
        return self.isp * G0


@dataclass
class LandingTarget:
    lat: float
    lon: float
    altitude: float = 0.0
    radius: float = 50.0


@dataclass
class BurnProfile:
    start_altitude: float
    duration: float
    throttle: float
    fuel_required: float
    deceleration: float
    method: str


@dataclass
class GuidanceCommand:
    throttle: float
    pitch_rate: float
    yaw_rate: float
    grid_fin_angle: float
    landing_burn_active: bool
    target_lat: float
    target_lon: float


class SuicideBurnCalculator:
    def __init__(self, gravity: float = G0, isp: float = 311.0):
        self.gravity = gravity
        self.isp = isp

    def burn_altitude(
        self, altitude: float, velocity: float, mass: float, thrust: float
    ) -> float:
        """Altitude at which to begin suicide burn for zero-velocity landing."""
        if velocity <= 0 or thrust <= mass * self.gravity:
            return 0.0

        exhaust_v = self.isp * G0
        mass_flow = thrust / exhaust_v
        a_net = thrust / mass - self.gravity

        if a_net <= 0:
            return float("inf")

        t_burn = velocity / a_net
        h_burn = velocity * t_burn / 2

        return max(0, altitude - h_burn)

    def throttle_for_deceleration(
        self, velocity: float, altitude: float, mass: float, thrust: float
    ) -> float:
        """Required throttle percentage to stop before ground."""
        if velocity <= 0 or altitude <= 0:
            return 0.0

        t_stop = 2 * altitude / velocity
        a_required = velocity / t_stop + self.gravity

        a_available = thrust / mass
        if a_available <= 0:
            return 1.0

        return min(1.0, a_required / a_available)

    def gravity_turn(
        self, altitude: float, max_altitude: float
    ) -> float:
        """Pitch angle during ascent gravity turn."""
        if max_altitude <= 0:
            return 90.0
        progress = altitude / max_altitude
        return 90.0 * math.exp(-2.5 * progress)

    def hoverslam_timing(
        self, altitude: float, velocity: float, mass: float, thrust: float
    ) -> Optional[BurnProfile]:
        """Compute hoverslam (suicide burn) parameters."""
        exhaust_v = self.isp * G0
        mass_flow = thrust / exhaust_v
        a_net = thrust / mass - self.gravity

        if a_net <= 0:
            return None

        t_burn = velocity / a_net
        h_burn = velocity * t_burn / 2

        fuel = mass_flow * t_burn
        start_alt = max(0, altitude - h_burn)

        if start_alt <= 0 and altitude > h_burn:
            return None

        return BurnProfile(
            start_altitude=start_alt,
            duration=t_burn,
            throttle=1.0,
            fuel_required=fuel,
            deceleration=a_net,
            method="HOVERSLAM",
        )


class GridFinController:
    def __init__(self, max_angle: float = 30.0, response_rate: float = 10.0):
        self.max_angle = max_angle
        self.response_rate = response_rate
        self._current_angle = 0.0

    def compute_deflection(
        self, current_heading: float, target_heading: float, velocity: float
    ) -> float:
        error = target_heading - current_heading
        k_p = 0.5
        k_d = 0.1

        command = k_p * error - k_d * current_heading
        return max(-self.max_angle, min(self.max_angle, command))

    def update(self, target_angle: float, dt: float) -> float:
        max_change = self.response_rate * dt
        delta = target_angle - self._current_angle
        if abs(delta) > max_change:
            delta = max_change if delta > 0 else -max_change
        self._current_angle += delta
        return self._current_angle


class LandingGuidance:
    def __init__(self, target: LandingTarget):
        self.target = target
        self.suicide_calc = SuicideBurnCalculator()
        self.grid_fin = GridFinController()
        self._burn_active = False
        self._descent_start: float = 0.0
        self._altitude_log: list[tuple[float, float]] = []

    def compute_command(
        self, state: BoosterState, dt: float = 0.1
    ) -> GuidanceCommand:
        self._altitude_log.append((time.time(), state.altitude))

        burn = self.suicide_calc.hoverslam_timing(
            state.altitude, state.velocity, state.mass, state.thrust
        )

        if burn and state.altitude <= burn.start_altitude * 1.05:
            self._burn_active = True
            throttle = self.suicide_calc.throttle_for_deceleration(
                state.velocity, state.altitude, state.mass, state.thrust
            )
        elif self._burn_active and state.velocity < 1.0:
            self._burn_active = False
            throttle = self.suicide_calc.throttle_for_deceleration(
                state.velocity, state.altitude, state.mass, state.thrust
            )
        else:
            throttle = 0.0
            self._burn_active = False

        heading_error = math.atan2(
            self.target.lon - state.lon, self.target.lat - state.lat
        )
        grid_angle = self.grid_fin.compute_deflection(
            state.yaw, math.degrees(heading_error), state.velocity
        )
        self.grid_fin.update(grid_angle, dt)

        return GuidanceCommand(
            throttle=max(0, min(1, throttle)),
            pitch_rate=0.0,
            yaw_rate=0.0,
            grid_fin_angle=self.grid_fin._current_angle,
            landing_burn_active=self._burn_active,
            target_lat=self.target.lat,
            target_lon=self.target.lon,
        )

    @property
    def descent_stats(self) -> dict:
        if not self._altitude_log:
            return {}
        first_alt = self._altitude_log[0][1]
        last_alt = self._altitude_log[-1][1]
        elapsed = self._altitude_log[-1][0] - self._altitude_log[0][0]
        return {
            "start_altitude": first_alt,
            "current_altitude": last_alt,
            "descent_time": round(elapsed, 1),
            "descent_rate": round((first_alt - last_alt) / max(elapsed, 0.01), 1),
            "burn_active": self._burn_active,
        }
