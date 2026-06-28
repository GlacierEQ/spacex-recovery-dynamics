"""Recovery dynamics tests."""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from alpha.landing_guidance import (
    BoosterState, LandingTarget, SuicideBurnCalculator, GridFinController,
    LandingGuidance, G0, MonteCarloLanding,
)
from omega.recovery_controller import (
    BargeController, BargePosition, BargeState, RecoveryCoordinator,
    WaveCondition, create_spacex_recovery,
)


def test_suicide_burn_altitude():
    calc = SuicideBurnCalculator()
    alt = calc.burn_altitude(5000, 300, 25000, 760000)
    assert alt > 0
    assert alt < 5000


def test_suicide_burn_no_thrust():
    calc = SuicideBurnCalculator()
    alt = calc.burn_altitude(1000, 300, 25000, 100000)
    assert alt == 0 or alt == float("inf")


def test_throttle_computation():
    calc = SuicideBurnCalculator()
    throttle = calc.throttle_for_deceleration(300, 500, 25000, 760000)
    assert 0 < throttle <= 1.0


def test_hoverslam_timing():
    calc = SuicideBurnCalculator()
    burn = calc.hoverslam_timing(1000, 300, 25000, 760000)
    assert burn is not None
    assert burn.duration > 0
    assert burn.fuel_required > 0
    assert burn.method == "HOVERSLAM"


def test_grid_fin_deflection():
    gf = GridFinController()
    angle = gf.compute_deflection(0, 45, 100)
    assert -30 <= angle <= 30


def test_grid_fin_update():
    gf = GridFinController()
    angle1 = gf.update(20, 0.1)
    assert angle1 > 0
    angle2 = gf.update(20, 0.1)
    assert angle2 > angle1


def test_landing_guidance():
    target = LandingTarget(28.5, -80.6)
    guidance = LandingGuidance(target)

    state = BoosterState(
        altitude=2000, velocity=200, mass=25000,
        thrust=760000, isp=311, drag_coeff=0.5, cross_section=10.0,
    )

    cmd = guidance.compute_command(state)
    assert 0 <= cmd.throttle <= 1.0
    assert cmd.target_lat == 28.5


def test_barge_transit():
    barge = BargeController("OCISLY", BargePosition(28.5, -80.6))
    assert barge.state == BargeState.TRANSIT

    target = BargePosition(30.0, -75.0)
    assert barge.transit_to(target)
    assert barge.state == BargeState.TRANSIT


def test_barge_station_keeping():
    barge = BargeController("OCISLY", BargePosition(28.5, -80.6))
    barge.transit_to(BargePosition(30.0, -75.0))
    barge.update_position(30.0, -75.0, 0, 0, 0.1)
    assert barge.state == BargeState.STATION_KEEPING


def test_barge_wave_compensation():
    barge = BargeController("OCISLY", BargePosition(28.5, -80.6))
    wave = WaveCondition(height_m=2.0, period_s=8.0)
    surge, sway = barge.compensate_waves(wave, 0.1)
    assert isinstance(surge, float)
    assert isinstance(sway, float)


def test_barge_wave_unsafe():
    barge = BargeController("OCISLY", BargePosition(28.5, -80.6))
    wave = WaveCondition(height_m=5.0, period_s=8.0)
    surge, sway = barge.compensate_waves(wave, 0.1)
    assert surge == 0.0
    assert sway == 0.0


def test_recovery_coordinator():
    coord = create_spacex_recovery()
    assert len(coord._barges) == 2
    coord.deploy_barge("OCISLY", BargePosition(30.0, -75.0))
    assert coord.fleet_status["OCISLY"]["state"] == "TRANSIT"


def test_landing_assignment():
    coord = create_spacex_recovery()
    coord.deploy_barge("OCISLY", BargePosition(28.5, -80.6))
    coord.deploy_barge("JRTI", BargePosition(32.0, -78.0))

    coord.get_barge("OCISLY").update_position(28.5, -80.6, 0, 0, 0.1)
    coord.get_barge("JRTI").update_position(32.0, -78.0, 0, 0, 0.1)

    barge_name = coord.assign_landing(1, 29.0, -79.0)
    assert barge_name in ("OCISLY", "JRTI")


def test_monte_carlo_landing():
    mc = MonteCarloLanding(seed=42, num_samples=200)
    result = mc.simulate(
        nominal_altitude=5000,
        nominal_velocity=300,
        nominal_mass=25000,
        thrust=760000,
        target_lat=28.5,
        target_lon=-80.6,
    )
    assert result["samples"] == 200
    assert 0 <= result["success_rate"] <= 1
    assert result["mean_distance_m"] >= 0
    assert "p95_distance_m" in result


def test_monte_carlo_deterministic():
    mc1 = MonteCarloLanding(seed=42, num_samples=100)
    r1 = mc1.simulate(5000, 300, 25000, 760000, 28.5, -80.6)

    mc2 = MonteCarloLanding(seed=42, num_samples=100)
    r2 = mc2.simulate(5000, 300, 25000, 760000, 28.5, -80.6)

    assert r1["mean_distance_m"] == r2["mean_distance_m"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
