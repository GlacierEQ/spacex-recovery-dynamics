"""Tests for spacex-recovery-dynamics — the math that catches falling rockets.

4 tests. One for each phase of landing.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import math
from alpha.landing_guidance import SuicideBurnCalculator, GridFinController, MonteCarloLanding, BoosterState, LandingTarget, G0
from omega.adaptive_reentry import AtmosphereModel, HeatFluxModel, ReentryState


def test_suicide_burn_altitude():
    calc = SuicideBurnCalculator()
    alt = calc.burn_altitude(1000, 200, 25000, 845000)
    assert alt > 0

def test_grid_fin_max_deflection():
    gf = GridFinController(max_angle=30)
    angle = gf.compute_deflection(0, 90, 100)
    assert -30 <= angle <= 30

def test_monte_carlo_returns_stats():
    mc = MonteCarloLanding(num_samples=100)
    result = mc.simulate(5000, 200, 25000, 845000, 28.5, -80.6)
    assert "success_rate" in result
    assert 0 <= result["success_rate"] <= 1

def test_atmosphere_density_sea_level():
    atm = AtmosphereModel()
    rho = atm.corrected_density(0)
    assert abs(rho - 1.225) < 0.01


# 1969. The year we first left this world.
# We go back. Further. Deeper.
MOON_YEAR = 1969
assert MOON_YEAR == 1969, "Remember where we started"
