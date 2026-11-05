"""Outcomes, the public site and the alert email."""

import json

import numpy as np
import pandas as pd
import pytest

from heatwave.pipeline import outcomes
from heatwave.publish import alerts, site

S = pd.Timestamp("2027-05-10")
NAMES = {"a": "Agra", "b": "<Banda>"}
REGIONS = {"a": "western", "b": "bundelkhand"}
SPEC = {
    "version": "m2_v1",
    "tiers": {
        "watch": {"threshold": 0.157},
        "alert": {"threshold": 0.272},
        "warning": {"threshold": 0.366},
    },
}


def forecast(status="ok", reason="", tier=("warning", "none"), p=(0.5, 0.1), asof="2027-05-07"):
    rows = []
    for d, ti, pi in zip("ab", tier, p, strict=True):
        for h in range(1, 11):
            rows.append(("lead", d, S + pd.Timedelta(days=h), pd.NaT, pi / 2, "none", True))
        rows.append(
            ("window", d, S + pd.Timedelta(days=1), S + pd.Timedelta(days=10), pi, ti, True)
        )
    t = pd.DataFrame(
        rows,
        columns=["target", "district_id", "valid_date", "window_end", "p", "tier", "in_season"],
    )
    return t.assign(
        issue_date=S,
        asof_date=pd.Timestamp(asof) if asof else pd.NaT,
        status=status,
        reason=reason,
        data_age_days=3,
        model_version="m2_v1",
        label_version="lbl",
        model_lead=0,
        p_clim=0.05,
    )


def labels(values: dict):
    return pd.DataFrame(
        [
            {"district_id": d, "obs_date": pd.Timestamp(day), "is_heatwave": v}
            for (d, day), v in values.items()
        ]
    )


def rows_for(out, target, district):
    return out[(out["target"] == target) & (out["district_id"] == district)]


def test_lead_outcomes_are_known_only_once_observed():
    out = outcomes.fill(forecast(), labels({("a", "2027-05-11"): True, ("a", "2027-05-12"): False}))
    a = rows_for(out, "lead", "a").set_index("valid_date")["observed"]
    assert a["2027-05-11"] == 1 and a["2027-05-12"] == 0
    assert np.isnan(a["2027-05-13"])


def test_a_window_is_decided_by_a_heatwave_and_otherwise_needs_every_day():
    days = pd.date_range("2027-05-11", "2027-05-20")
    seen = dict.fromkeys([("a", d) for d in days[:3]], False)
    assert np.isnan(
        rows_for(outcomes.fill(forecast(), labels(seen)), "window", "a")["observed"].iloc[0]
    )
    seen[("a", days[1])] = True
    assert rows_for(outcomes.fill(forecast(), labels(seen)), "window", "a")["observed"].iloc[0] == 1
    all_quiet = dict.fromkeys([("a", d) for d in days], False)
    assert (
        rows_for(outcomes.fill(forecast(), labels(all_quiet)), "window", "a")["observed"].iloc[0]
        == 0
    )


def test_outcomes_run_rewrites_the_year(tmp_path):
    path = tmp_path / "forecasts" / "issue_year=2027" / "2027-05-10.parquet"
    path.parent.mkdir(parents=True)
    forecast().to_parquet(path, index=False)
    lab = labels({("a", "2027-05-11"): True})
    assert outcomes.run(tmp_path, lab) == [tmp_path / "outcomes" / "issue_year=2027.parquet"]
    out = pd.read_parquet(tmp_path / "outcomes" / "issue_year=2027.parquet")
    assert out["observed"].sum() == 2  # lead day and window for district a


def test_every_page_says_it_is_not_official():
    for page in (site.index_html(forecast(), NAMES, REGIONS), site.about_html(SPEC, "https://x")):
        assert "not an official forecast or warning" in page
        assert "mausam.imd.gov.in" in page


def test_about_credits_sources_and_says_imd_has_not_endorsed():
    page = site.about_html(SPEC, "https://x")
    for needed in ("NASA", "asl.232", "has not reviewed or endorsed", "15.7%", "27.2%", "36.6%"):
        assert needed in page


def test_index_sorts_by_tier_and_escapes_names():
    page = site.index_html(forecast(), NAMES, REGIONS)
    assert "&lt;Banda&gt;" in page and "<Banda>" not in page
    assert page.index("Agra") < page.index("Banda")
    assert 't-warning">Warning' in page
    assert "Warning: 1, Alert: 0, Watch: 0" in page


def test_degraded_says_so():
    t = forecast("degraded", "data 3 days late", ("none", "none"), asof=None)
    page = site.index_html(t, NAMES, REGIONS)
    assert "Degraded." in page and "data 3 days late" in page and "climatology only" in page


def test_late_data_says_so():
    line = site.status_line(forecast("late_1", "data 1 day late"))
    assert "Data 1 day late" in line and "Fewer days" in line


def test_out_of_season_shows_dashes_not_numbers():
    t = forecast(tier=("none", "none"), p=(np.nan, np.nan)).assign(in_season=False)
    page = site.index_html(t, NAMES, REGIONS)
    assert (
        "Out of season" in page and "–" in page and "nan" not in page.lower().replace("finance", "")
    )


def test_build_writes_the_site_and_nothing_from_imd(tmp_path):
    f = tmp_path / "f" / "issue_year=2027"
    f.mkdir(parents=True)
    forecast().to_parquet(f / "2027-05-10.parquet", index=False)
    out = tmp_path / "site"
    site.build(tmp_path / "f", out, NAMES, REGIONS, SPEC, "https://x", 30)
    for name in ("index.html", "about.html", ".nojekyll", "data/latest.csv", "data/latest.json"):
        assert (out / name).exists()
    table = pd.read_csv(out / "data" / "latest.csv")
    assert "label_version" not in table.columns
    assert not [c for c in table.columns if "imd" in c.lower()]
    assert len(json.loads((out / "data" / "latest.json").read_text())) == len(table)
    assert json.loads((out / "data" / "index.json").read_text())["days"] == ["2027-05-10.csv"]


class FakeSmtp:
    sent: list = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def login(self, user, password):
        self.user = user

    def send_message(self, msg):
        FakeSmtp.sent.append(msg)


@pytest.fixture
def smtp(monkeypatch):
    FakeSmtp.sent = []
    monkeypatch.setenv("ALERT_SMTP_USER", "from@example.com")
    monkeypatch.setenv("ALERT_SMTP_PASSWORD", "x")
    monkeypatch.setenv("ALERT_TO", "to@example.com")
    return FakeSmtp


def test_warning_sends_one_email_per_day(tmp_path, smtp):
    args = (forecast(), NAMES, {}, "https://site", tmp_path)
    assert alerts.send(*args, smtp=smtp) == "sent"
    assert alerts.send(*args, smtp=smtp) == "already sent"
    assert len(smtp.sent) == 1
    msg = smtp.sent[0]
    assert msg["To"] == "to@example.com"
    body = msg.get_content()
    assert "Agra: 50%" in body and "Not an official warning" in body
    assert "Banda" not in body


def test_no_warning_no_email(tmp_path, smtp):
    t = forecast(tier=("alert", "none"))
    assert alerts.send(t, NAMES, {}, "https://site", tmp_path, smtp=smtp) == "none"
    assert not smtp.sent


def test_without_secrets_nothing_is_sent_and_it_can_retry(tmp_path, monkeypatch):
    for k in ("ALERT_SMTP_USER", "ALERT_SMTP_PASSWORD", "ALERT_TO"):
        monkeypatch.delenv(k, raising=False)
    assert alerts.send(forecast(), NAMES, {}, "https://site", tmp_path) == "not configured"
    assert not list(tmp_path.iterdir())


def test_the_test_email_says_it_is_a_test(smtp):
    alerts.send_test({}, "https://site", smtp=smtp)
    msg = smtp.sent[0]
    assert msg["To"] == "to@example.com" and "test" in msg["Subject"]
    assert "no heatwave is being reported" in msg.get_content()


def test_the_test_email_names_a_missing_secret_without_any_value(monkeypatch, smtp):
    monkeypatch.delenv("ALERT_TO")
    with pytest.raises(SystemExit, match="ALERT_TO") as e:
        alerts.send_test({}, "https://site", smtp=smtp)
    assert "from@example.com" not in str(e.value)
    assert not smtp.sent
