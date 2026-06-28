"""Recovery controller — orchestrates ASDS barge operations and drone ship positioning.

Manages autonomous spaceport drone ship (ASDS) positioning, station-keeping,
and booster touchdown sequence. Handles wave compensation and wind corrections.
"""

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable

from alpha.landing_guidance import LandingTarget, BoosterState, GuidanceCommand


class BargeState(Enum):
    TRANSIT = auto()
    STATION_KEEPING = auto()
    HOLDING = auto()
    RECOVERY = auto()
    RETURNING = auto()


@dataclass
class BargePosition:
    lat: float
    lon: float
    heading: float = 0.0
    speed_knots: float = 0.0


@dataclass
class WaveCondition:
    height_m: float
    period_s: float
    direction_deg: float = 0.0

    @property
    def is_safe(self) -> bool:
        return self.height_m < 3.0


@dataclass
class RecoveryEvent:
    time: float
    event: str
    details: dict = field(default_factory=dict)


class BargeController:
    def __init__(self, name: str, home_port: BargePosition):
        self.name = name
        self.home_port = home_port
        self.state = BargeState.TRANSIT
        self._position = BargePosition(home_port.lat, home_port.lon)
        self._target_position: Optional[BargePosition] = None
        self._event_log: list[RecoveryEvent] = []
        self._callbacks: list[Callable] = []
        self._station_keep_tolerance = 0.0001

    def on_event(self, callback: Callable):
        self._callbacks.append(callback)

    def transit_to(self, target: BargePosition) -> bool:
        if self.state == BargeState.STATION_KEEPING:
            return False
        self._target_position = target
        self.state = BargeState.TRANSIT
        self._log("transit_started", {"target": (target.lat, target.lon)})
        return True

    def update_position(
        self, lat: float, lon: float, heading: float, speed: float, dt: float
    ) -> BargePosition:
        self._position = BargePosition(lat, lon, heading, speed)

        if self.state == BargeState.TRANSIT and self._target_position:
            dist = self._haversine(lat, lon, self._target_position.lat, self._target_position.lon)
            if dist < 100:
                self.state = BargeState.STATION_KEEPING
                self._log("station_keeping_entered", {"distance": dist})

        if self.state == BargeState.STATION_KEEPING:
            self._station_keep(dt)

        return self._position

    def _station_keep(self, dt: float):
        if not self._target_position:
            return

        dlat = self._target_position.lat - self._position.lat
        dlon = self._target_position.lon - self._position.lon

        k_p = 2.0
        correction_lat = k_p * dlat * dt
        correction_lon = k_p * dlon * dt

        self._position.lat += correction_lat
        self._position.lon += correction_lon

    def _haversine(self, lat1, lon1, lat2, lon2) -> float:
        R = 6371000
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        return R * 2 * math.asin(math.sqrt(a))

    def compensate_waves(self, wave: WaveCondition, dt: float) -> tuple[float, float]:
        if not wave.is_safe:
            return (0.0, 0.0)

        period = wave.period_s
        amp = wave.height_m / 2
        dir_rad = math.radians(wave.direction_deg)

        surge = amp * math.sin(2 * math.pi * time.time() / period)
        sway = amp * math.cos(2 * math.pi * time.time() / period)

        return (
            -surge * math.cos(dir_rad) * 0.1,
            -sway * math.sin(dir_rad) * 0.1,
        )

    def receive_booster(self, command: GuidanceCommand) -> dict:
        self.state = BargeState.RECOVERY
        self._log("booster_received", {
            "target": (command.target_lat, command.target_lon),
            "throttle": command.throttle,
        })

        return {
            "barge": self.name,
            "state": "RECOVERY",
            "position": (self._position.lat, self._position.lon),
            "target": (command.target_lat, command.target_lon),
        }

    def return_to_port(self) -> bool:
        self.state = BargeState.RETURNING
        self._target_position = self.home_port
        self._log("returning_to_port", {"home": (self.home_port.lat, self.home_port.lon)})
        return True

    def _log(self, event: str, details: dict):
        entry = RecoveryEvent(time=time.time(), event=event, details=details)
        self._event_log.append(entry)
        for cb in self._callbacks:
            cb(entry)

    @property
    def position(self) -> BargePosition:
        return self._position

    @property
    def event_log(self) -> list[dict]:
        return [{"time": e.time, "event": e.event, "details": e.details} for e in self._event_log]


class RecoveryCoordinator:
    def __init__(self):
        self._barges: dict[str, BargeController] = {}
        self._recovery_log: list[dict] = []

    def add_barge(self, barge: BargeController):
        self._barges[barge.name] = barge

    def deploy_barge(self, name: str, target: BargePosition) -> bool:
        barge = self._barges.get(name)
        if not barge:
            return False
        return barge.transit_to(target)

    def get_barge(self, name: str) -> Optional[BargeController]:
        return self._barges.get(name)

    def assign_landing(
        self, booster_id: int, target_lat: float, target_lon: float
    ) -> Optional[str]:
        available = [
            (name, barge) for name, barge in self._barges.items()
            if barge.state in (BargeState.STATION_KEEPING, BargeState.HOLDING)
        ]

        if not available:
            return None

        available.sort(key=lambda x: self._haversine(
            x[1].position.lat, x[1].position.lon, target_lat, target_lon
        ))

        best_name, best_barge = available[0]
        self._recovery_log.append({
            "booster": booster_id,
            "barge": best_name,
            "target": (target_lat, target_lon),
        })
        return best_name

    def _haversine(self, lat1, lon1, lat2, lon2) -> float:
        R = 6371000
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        return R * 2 * math.asin(math.sqrt(a))

    @property
    def fleet_status(self) -> dict:
        return {
            name: {"state": barge.state.name, "position": (barge.position.lat, barge.position.lon)}
            for name, barge in self._barges.items()
        }


def create_spacex_recovery() -> RecoveryCoordinator:
    coord = RecoveryCoordinator()

    ocesy = BargeController(
        "OCISLY",
        BargePosition(28.5, -80.6),
    )
    jrti = BargeController(
        "JRTI",
        BargePosition(32.0, -78.0),
    )

    coord.add_barge(ocesy)
    coord.add_barge(jrti)

    return coord
