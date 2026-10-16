"""Recompute the Sona v1 report's numbers from the per-question results in results/.

  python score.py
"""
from __future__ import annotations

import pandas as pd

from musp_eval.questions import SUBSETS, half
from musp_eval.stats import mcnemar_exact, wilson

N = {"P": 106, "general": 36, "tonality/style": 48, "composer/title": 48}
# MuSP-Bench Table 1, audio input (percent); no per-question data is published for these.
REPORTED = {
    "Muse Spark 1.2 (reported in [1])": {"P": 3.8, "general": 11.1, "tonality/style": 20.8, "composer/title": 2.1},
    "Audio Flamingo Next (reported in [1])": {"P": 0.9, "general": 0.0, "tonality/style": 22.9, "composer/title": 0.0},
}
PAIRS = [("Gemini 3.8 Flash", "gemini-3.8-flash_audio.csv", "gemini-3.8-flash_sona.csv"),
         ("Qwen3.5-Omni-Plus", "qwen3.5-omni-plus_audio.csv", "qwen3.5-omni-plus_sona.csv")]


def main() -> None:
    systems = pd.read_csv("results/systems.csv")
    res = {r.file: pd.read_csv(f"results/{r.file}") for r in systems.itertuples()}

    rows = []
    for r in systems.itertuples():
        g = res[r.file]
        assert len(g) == 238 and g.id.is_unique, r.file
        k = int(g.correct.sum())
        lo, hi = wilson(k, 238)
        rows.append({"system": r.system, **{s: 100 * g[g.subset == s].correct.mean() for s in SUBSETS},
                     "all": 100 * k / 238, "95% CI": f"{lo:.1f}-{hi:.1f}"})
    for name, pct in REPORTED.items():
        k = sum(round(pct[s] * N[s] / 100) for s in SUBSETS)
        rows.append({"system": name, **pct, "all": 100 * k / 238, "95% CI": ""})
    table = pd.DataFrame(rows).sort_values("all", ascending=False)
    print("Accuracy (%), strict grader, 238 audio-only questions\n")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    print("\nSame model, audio input vs audio input + Sona (paired, exact McNemar)\n")
    for model, a, c in PAIRS:
        A, C = res[a].set_index("id"), res[c].set_index("id").loc[res[a].id]
        for label, m in (("all", A.index), ("P", A.index[A.subset == "P"])):
            x, y = A.loc[m].correct, C.loc[m].correct
            only_a, only_c = int((x & ~y).sum()), int((~x & y).sum())
            print(f"  {model:<18} {label:<4} {100 * x.mean():5.1f} -> {100 * y.mean():5.1f}   "
                  f"only audio {only_a:3d}, only + Sona {only_c:3d}, p = {mcnemar_exact(only_a, only_c):.2g}")

    print("\nDevelopment half (odd-numbered pieces) vs held-out half (even-numbered pieces), accuracy (%)\n")
    out = []
    for r in systems.itertuples():
        g = res[r.file].assign(half=lambda d: d.piece.map(half))
        row = {"system": r.system}
        for h in ("development", "held-out"):
            d = g[g.half == h]
            row[f"{h} all ({len(d)})"] = 100 * d.correct.mean()
            p = d[d.subset == "P"]
            row[f"{h} P ({len(p)})"] = 100 * p.correct.mean()
        out.append(row)
    print(pd.DataFrame(out).to_string(index=False, float_format=lambda x: f"{x:.1f}"))


if __name__ == "__main__":
    main()
