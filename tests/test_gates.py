"""The realism gate set must be able to fail.

Before R0c the harness computed 36 statistics and gated 9 of them, so a regression in any
of the other 27 was silent, and the `octid` holdout gated 0 of 0 metrics -- it printed
"0/0 passing" and could never contribute to a non-zero exit code. "9/9" therefore meant
"9 gated metrics pass", not "the simulator matches".

These tests pin the machinery, not the tolerances: that every gated key is a real metric
name, that every waiver points at a gated metric, that each tier can actually fail, and
that a waived gate is reported without being counted as a pass.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from octsim import realism_metrics as rmet  # noqa: E402


#: The band names `measure` fills in. `band_above_floor_db` has no `vitreous` entry -- the
#: vitreous IS the floor, so that ratio is degenerate and the key is never emitted.
BANDS = ("vitreous", "inner", "onl", "rpe_complex", "choroid")


def _metric_names() -> set[str]:
    """Every key `aggregate` can emit, including the dotted band sub-keys.

    Built from a POPULATED metrics object: the dict-valued fields default to empty, so
    aggregating a default instance silently omits every `band_db.*` key and would make a
    real gate look like a typo.
    """
    m = rmet.RealismMetrics()
    m.band_db = {b: 0.0 for b in BANDS}
    m.band_above_floor_db = {b: 0.0 for b in BANDS if b != "vitreous"}
    return set(rmet.aggregate([m, m]))


def test_every_gate_names_a_metric_that_exists():
    """`compare` silently skips a key missing from either side, so a typo is a dead gate."""
    names = _metric_names()
    unknown = sorted(k for k in rmet.GATES if k not in names)
    assert not unknown, unknown


def test_every_waiver_points_at_a_gated_metric():
    """A waiver on an ungated metric is dead weight and hides nothing."""
    bad = sorted(f"{c}:{m}" for (c, m) in rmet.EXPECTED_FAIL if m not in rmet.GATES)
    assert not bad, bad


def test_every_waiver_gives_a_reason():
    for key, why in rmet.EXPECTED_FAIL.items():
        assert isinstance(why, str) and len(why) > 40, key


def test_each_tier_can_fail():
    """A tier with no gated metric is decoration: it can never report a problem.

    This is what `octid` was -- `TIER_COMPRESSED` kept four metrics and none was gated.
    """
    for tier_name in ("compressed", "display", "full"):
        tier = rmet.TIERS[tier_name]
        gated = {k for k in rmet.GATES if tier is None or k.split(".")[0] in tier}
        assert gated, f"tier {tier_name} gates nothing"


def test_compressed_tier_gates_something_that_survives_the_harness_drops():
    """`validate_realism` drops thickness (pitch-calibrated) and sag (FOV unknown) for octid.

    So a compressed-tier gate only bites if it is one of the two metrics left standing.
    """
    survives = {"rpe_peak_separation_um", "shadow_median_width_um"}
    assert survives & set(rmet.GATES), (
        "no compressed-tier gate survives the octid drop set, so octid gates 0/0 again")


def test_gate_outcome_reports_waivers_without_counting_them_as_passes():
    corpus, metric = next(iter(rmet.EXPECTED_FAIL))
    assert rmet.gate_outcome(False, corpus, metric) == "XFAIL"
    assert rmet.gate_outcome(True, corpus, metric) == "XPASS"
    # ...and an unwaived metric is unaffected
    assert rmet.gate_outcome(False, "training", "sag_optical_um") == "FAIL"
    assert rmet.gate_outcome(True, "training", "sag_optical_um") == "PASS"
    assert rmet.gate_outcome(None, "training", "whatever") is None


def test_sag_gate_can_resolve_the_inter_device_difference():
    """Real sag is 503 um on Maestro3 and 254 um on Maestro2.

    A tolerance wider than that 249 um gap cannot tell a correct render from one using the
    wrong device's scan-distortion correction, which is the single thing it exists to catch.
    """
    assert rmet.GATES["sag_optical_um"] < 249.0 / 2


def test_gate_coverage_is_more_than_a_quarter_of_the_vector():
    names = _metric_names()
    assert len(rmet.GATES) >= len(names) // 4, (len(rmet.GATES), len(names))


def test_format_table_marks_waived_rows():
    corpus, metric = next(iter(rmet.EXPECTED_FAIL))
    rows = [(metric, 1.0, 0.1, 9.0, 0.1, 8.0, rmet.GATES[metric], False)]
    assert "XFAIL" in rmet.format_table(rows, corpus)
    # the same row on a corpus with no waiver is a hard failure
    assert "FAIL" in rmet.format_table(rows, "some-other-corpus")
    assert "XFAIL" not in rmet.format_table(rows, "some-other-corpus")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} gate-coverage checks passed")
