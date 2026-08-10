"""Smoke tests for ``vol_tracker_charts`` -- each figure renders to a
non-empty base64 PNG in both themes from tiny synthetic inputs, and empty
inputs are skipped rather than crashed on (the page must always build)."""
import numpy as np

import vol_tracker_charts as C


def _paths():
    med = {s: list(np.linspace(0, m, 127))
           for s, m in (("QQQ", 8.0), ("GDX", 5.0), ("XLE", 4.0),
                        ("SMH", 12.0), ("IWM", 6.0))}
    up = {s: {"med63": 3.0, "hit63": 65.0, "mae_q25": -7.0, "cls": "CHASER"}
          for s in med}
    up["GDX"] = {"med63": -0.5, "hit63": 48.0, "mae_q25": -16.0,
                 "cls": "ROUND-TRIP"}
    dn = {s: {"med63": 5.0, "hit63": 70.0, "n": 60} for s in med}
    turn = {"SMH": {"med63": 11.0, "hit63": 100.0, "n": 8},
            "GDX": {"med63": 1.3, "hit63": 62.0, "n": 13},
            "XLF": {"med63": -7.8, "hit63": 44.0, "n": 9},
            "XLP": {"med63": 2.0, "hit63": 50.0, "n": 3}}   # n<8 -> dropped
    return {"med_paths": med, "up": up, "dn": dn, "turn": turn}


def _exp_rows():
    return [{"symbol": s, "cond63": 7.0 + i, "cond126": 11.0 + i,
             "imp3m": 9.0 + i, "imp6m": 13.0 + i}
            for i, s in enumerate(["QQQ", "GDX", "SMH", "IWM", "XLE", "XLF"])]


def _skew_rows():
    return [{"symbol": s, "rr": r, "upside": u}
            for s, r, u in (("GDX", 4.3, 31), ("QQQ", -4.0, 34),
                            ("SMH", -3.8, 37), ("IWM", -4.3, 67),
                            ("EEM", 1.7, 6))]


def test_render_all_both_themes():
    out = C.render_all(_paths(), _exp_rows(), _skew_rows())
    assert set(out) == {"paths", "expmove", "skew"}
    for key, imgs in out.items():
        assert set(imgs) == {"light", "dark"}
        for b64 in imgs.values():
            assert isinstance(b64, str) and len(b64) > 10_000


def test_empty_inputs_are_skipped():
    out = C.render_all({"med_paths": {}, "up": {}, "dn": {}, "turn": {}},
                       [], [])
    assert out == {}


def test_thin_inputs_are_skipped():
    # fewer than 5 usable rows -> no scatter figures
    assert C.fig_expmove(_exp_rows()[:3], C.THEMES["light"]) is None
    assert C.fig_skew(_skew_rows()[:3], C.THEMES["light"]) is None
