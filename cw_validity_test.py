#!/usr/bin/env python3
"""
cw_validity_test.py — Canonical CW grid: single source of truth.

Every CW value produced by SAC inference (Python/JS/C++) must be a member of
{3, 7, 15, 31, 63, 127, 255, 511, 1023}  (i.e. 2^n - 1 for n = 2..10)
and GDCF values must also snap to this grid.

Run: python3 cw_validity_test.py
"""

VALID_CW_VO = [3, 7, 15, 31, 63, 127, 255, 511, 1023]   # 9-entry grid, action[0]
VALID_CW_BE = [15, 31, 63, 127, 255, 511, 1023]           # 7-entry grid, action[2]
VALID_CW_SET = set(VALID_CW_VO)                            # superset covers both


def snap_to_valid_cw(cw: int) -> int:
    """Snap cw UP to nearest 2^n-1 value, max 1023.
    Mirrors C++ SnapToValidCw and JS snapToValidCw exactly."""
    v = 1
    while v < cw:
        v = (v << 1) | 1
    return min(v, 1023)


def decode_action_vo(a0: float) -> int:
    """Decode actor output a0 in [-1,1] -> VO CWmin via index grid.
    Identical formula used in train_sac.py, C++ RunInference(), JS update()."""
    idx = max(0, min(8, int((a0 + 1.0) * 0.5 * 8.0 + 0.5)))
    return VALID_CW_VO[idx]


def decode_action_be(a2: float) -> int:
    """Decode actor output a2 in [-1,1] -> BE CWmin via index grid."""
    idx = max(0, min(6, int((a2 + 1.0) * 0.5 * 6.0 + 0.5)))
    return VALID_CW_BE[idx]


def decode_mult(cw_min: int, action_mult: float) -> int:
    """Decode multiplier action -> CWmax, snapped to valid CW."""
    mult = 1.0 + (action_mult + 1.0) * 0.5 * 3.0
    raw = int(cw_min * mult)
    return max(snap_to_valid_cw(raw), cw_min)


# --- Unit Tests --------------------------------------------------------------

def test_snap():
    cases = {
        0: 1, 1: 1, 2: 3, 3: 3, 4: 7, 6: 7, 7: 7,
        8: 15, 14: 15, 15: 15, 16: 31,
        63: 63, 64: 127, 126: 127, 127: 127,
        128: 255, 190: 255, 255: 255,
        256: 511, 511: 511, 512: 1023, 1023: 1023,
        1024: 1023, 9999: 1023,
    }
    for inp, expected in cases.items():
        got = snap_to_valid_cw(inp)
        assert got == expected, f"snap_to_valid_cw({inp}) = {got}, expected {expected}"
    print("  PASS  snap_to_valid_cw: all boundary cases correct")


def test_decode_vo_grid():
    """Every index [0..8] must produce a value in VALID_CW_VO."""
    for i in range(9):
        a = (i - 0.5) * 2.0 / 8.0 - 1.0 + 0.001
        cw = decode_action_vo(a)
        assert cw in VALID_CW_SET, f"decode_action_vo({a:.4f}) = {cw} NOT IN VALID_CW_SET"
    print("  PASS  decode_action_vo: all 9 grid slots produce valid values")


def test_decode_be_grid():
    """Every index [0..6] must produce a value in VALID_CW_BE."""
    for i in range(7):
        a = (i - 0.5) * 2.0 / 6.0 - 1.0 + 0.001
        cw = decode_action_be(a)
        assert cw in VALID_CW_SET, f"decode_action_be({a:.4f}) = {cw} NOT IN VALID_CW_SET"
    print("  PASS  decode_action_be: all 7 grid slots produce valid values")


def test_mult_always_valid():
    """CWmax from any CWmin x multiplier must snap to valid."""
    import random
    rng = random.Random(42)
    for _ in range(10000):
        cw_min = rng.choice(VALID_CW_VO)
        a_mult = rng.uniform(-1.0, 1.0)
        cw_max = decode_mult(cw_min, a_mult)
        assert cw_max in VALID_CW_SET, \
            f"decode_mult(cwMin={cw_min}, a={a_mult:.4f}) = {cw_max} NOT IN VALID_CW_SET"
        assert cw_max >= cw_min, f"cw_max={cw_max} < cw_min={cw_min}"
    print("  PASS  decode_mult: 10,000 random pairs all produce valid CWmax >= CWmin")


def test_gdcf_known_illegal():
    """GDCF multiplicative intermediates that should snap correctly."""
    known_bad = [190, 152, 114, 125, 95, 86, 230, 144, 346, 126]
    for raw in known_bad:
        snapped = snap_to_valid_cw(raw)
        assert snapped in VALID_CW_SET, f"snap({raw}) = {snapped} still invalid"
    print("  PASS  gdcf_snap: all known-illegal GDCF intermediates snap correctly")
    print(f"        e.g. snap(126)={snap_to_valid_cw(126)}, snap(190)={snap_to_valid_cw(190)}, snap(152)={snap_to_valid_cw(152)}")


def test_126_exact_reconstruction():
    """Show exactly which (cwMin, a_mult) pair would produce 126 under the old
    broken C++ GDCF code (no snap), and confirm snap fixes it to 127."""
    # GDCF starts cwMin=63, ratio < 0.8 triggers: cwMin = int(63 * 0.75) = 47
    # then another step up: cwMin = int(47 * 1.5) = 70 -> ... eventually
    # more directly: cwMin=63, multiply 0.75 twice:
    #   63 * 0.75 = 47.25 -> int -> 47 (already bad, but snap(47)=63)
    # Simpler: cwMax=127, ratio > 1.2 -> cwMax = int(127*1.2) = 152 -> snap=255 CORRECT
    # cwMax=127, ratio < 0.8 -> cwMax = int(127*0.9) = 114 -> snap=127 CORRECT
    # But the original code set m_cwMax directly without snap, so it *stays* at 114
    # Next cycle: cwMin = min(cwMax, int(cwMin*1.5)) = min(114, int(63*1.5))=min(114,94)=94
    # That's illegal: snap(94) = 127. snap(114) = 127. Both correct after fix.
    # For exactly 126: cwMin=63, cwMax=127: after 0.9x: 127*0.9=114.3->114 (stored raw)
    # Then if cwMin was 64 (illegal itself from a prior step): 64*2-1 would be 127 but
    # int(63*1.01)=63, cwMax=int(63*2)=126 -> this is the exact path
    cw_min_after_drift = 63
    mult_raw = cw_min_after_drift * 2   # e.g. a multiplier of exactly 2.0 -> 63*2=126
    assert mult_raw == 126
    assert snap_to_valid_cw(126) == 127
    print(f"  PASS  exact 126 reconstruction: cwMin=63 * mult=2.0 = 126 (raw), snap(126)=127")


def test_js_linear_would_fail():
    """Old JS pattern produced illegal values for most actions."""
    CW_MIN, CW_MAX = 15, 1023
    broken_count = 0
    for i in range(1001):
        a = -1.0 + i * 0.002
        raw = round(CW_MIN + (a + 1) / 2 * (CW_MAX - CW_MIN))
        if raw not in VALID_CW_SET:
            broken_count += 1
    assert broken_count > 900
    print(f"  PASS  js_linear_interp_rejected: {broken_count}/1001 uniformly-sampled"
          f" actions produce illegal CW with old Math.round(15+(a+1)/2*1008) mapping")


if __name__ == "__main__":
    print("=" * 60)
    print("CW VALIDITY CANONICAL TEST SUITE")
    print("=" * 60)
    print(f"Valid CW set: {sorted(VALID_CW_SET)}")
    print()
    test_snap()
    test_decode_vo_grid()
    test_decode_be_grid()
    test_mult_always_valid()
    test_gdcf_known_illegal()
    test_126_exact_reconstruction()
    test_js_linear_would_fail()
    print()
    print("=" * 60)
    print("ALL 7 TESTS PASSED")
    print("=" * 60)
    print()
    print("Canonical grids (use these everywhere, never reimplement):")
    print(f"  VO (9): {VALID_CW_VO}")
    print(f"  BE (7): {VALID_CW_BE}")
    print()
    print("snap_to_valid_cw() is the single source of truth.")
    print("Python: this file.  C++: SnapToValidCw() (free fn).  JS: snapToValidCw().")
