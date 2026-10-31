"""Gate G3: does LightGBM (M3) beat logistic regression (M2)? (reference.md §3.7, §6.1)

M3 goes live only if, on the ONSET subset, at some lead, BSS(M3) - BSS(M2) has a
paired year-block interval entirely above 0. Otherwise M2 goes live.
Both models are scored on identical OOF rows and identical resampled years.
"""

from __future__ import annotations

import pandas as pd

from heatwave.validation import metrics


def g3(lead_oof: pd.DataFrame, cfg: dict, challenger: str = "M3", champion: str = "M2") -> dict:
    e = cfg["evaluation"]
    t = metrics.bss_table(
        lead_oof[lead_oof["onset"]],
        [challenger, champion],
        "M0",
        "lead",
        e["bootstrap_resamples"],
        e["bootstrap_seed"],
        e["ci"],
        pairs=[(challenger, champion)],
    )
    d = t[t["model"] == f"{challenger}-{champion}"].sort_values("group")
    wins = d[d["lo"] > 0]["group"].astype(int).tolist()
    return {"passed": bool(wins), "leads_won": wins, "table": d}


def comparison_tables(
    lead_oof: pd.DataFrame,
    window_oof: pd.DataFrame | None,
    cfg: dict,
    models: list[str],
    pairs: list[tuple[str, str]],
) -> dict[str, pd.DataFrame]:
    e = cfg["evaluation"]
    args = (e["bootstrap_resamples"], e["bootstrap_seed"], e["ci"])
    out = {
        "lead_all": metrics.bss_table(lead_oof, models, "M0", "lead", *args, pairs=pairs),
        "lead_onset": metrics.bss_table(
            lead_oof[lead_oof["onset"]], models, "M0", "lead", *args, pairs=pairs
        ),
        "region": metrics.bss_table(lead_oof, models, "M0", "region", *args, pairs=pairs),
        "month": metrics.bss_table(lead_oof, models, "M0", "month", *args, pairs=pairs),
    }
    if window_oof is not None:
        out["window_all"] = metrics.bss_table(window_oof, models, "M0", None, *args, pairs=pairs)
        out["window_onset"] = metrics.bss_table(
            window_oof[window_oof["onset"]], models, "M0", None, *args, pairs=pairs
        )
    return out
