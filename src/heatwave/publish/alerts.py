"""Email when a district reaches Warning (reference.md §5.2, docs/decisions/012).

The account and recipient come from environment variables (GitHub secrets), never from the
public repo. One email per issue date: a marker stops a same-day re-run sending it again.
Pipeline failures are reported by GitHub's own failed-run email.
"""

from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

import pandas as pd


def warnings_today(rows: pd.DataFrame) -> pd.DataFrame:
    w = rows[(rows["target"] == "window") & (rows["tier"] == "warning")]
    return w.sort_values("p", ascending=False)


def compose(rows: pd.DataFrame, names: dict, site_url: str, sender: str, to: str) -> EmailMessage:
    w = warnings_today(rows)
    issue = pd.Timestamp(rows["issue_date"].iloc[0]).date()
    first, last = w["valid_date"].iloc[0], w["window_end"].iloc[0]
    lines = [
        f"{names.get(d, d)}: {100 * p:.0f}%" for d, p in zip(w["district_id"], w["p"], strict=True)
    ]
    status = rows["reason"].iloc[0]
    msg = EmailMessage()
    msg["Subject"] = f"Heatwave model: Warning tier in {len(w)} districts ({issue})"
    msg["From"], msg["To"] = sender, to
    msg.set_content(
        f"Chance of a heatwave day {pd.Timestamp(first):%d %b} to {pd.Timestamp(last):%d %b}:\n\n"
        + "\n".join(lines)
        + (f"\n\nNote: {status}." if status else "")
        + f"\n\n{site_url}\n\nModel tier from an experimental forecast. Not an official warning."
    )
    return msg


def _account() -> tuple[str | None, str | None, str | None]:
    return (
        os.environ.get("ALERT_SMTP_USER"),
        os.environ.get("ALERT_SMTP_PASSWORD"),
        os.environ.get("ALERT_TO"),
    )


def _deliver(msg: EmailMessage, user: str, password: str, cfg: dict, smtp=None) -> None:
    connect = smtp or (lambda: smtplib.SMTP_SSL(cfg["smtp_host"], cfg["smtp_port"], timeout=60))
    with connect() as server:
        server.login(user, password)
        server.send_message(msg)


def send_test(cfg: dict, site_url: str, smtp=None) -> None:
    """One test email. Raises if the secrets are missing or sending fails. Prints no
    account details: the workflow log of a public repo is public."""
    user, password, to = _account()
    missing = [
        name
        for name, value in (
            ("ALERT_SMTP_USER", user),
            ("ALERT_SMTP_PASSWORD", password),
            ("ALERT_TO", to),
        )
        if not value
    ]
    if missing:
        raise SystemExit("missing secrets: " + ", ".join(missing))
    msg = EmailMessage()
    msg["Subject"] = "Heatwave model: test email"
    msg["From"], msg["To"] = user, to
    msg.set_content(
        "This is a test. Your alert setup works, and no heatwave is being reported.\n\n"
        "Real alerts are sent when a district reaches the model's Warning tier, which can "
        f"only happen from about March to July.\n\n{site_url}\n"
    )
    _deliver(msg, user, password, cfg, smtp)


def send(
    rows: pd.DataFrame, names: dict, cfg: dict, site_url: str, marker_dir: Path, smtp=None
) -> str:
    """Returns what happened: 'none', 'already sent', 'not configured' or 'sent'."""
    if warnings_today(rows).empty:
        return "none"
    issue = pd.Timestamp(rows["issue_date"].iloc[0]).date()
    marker = marker_dir / f"{issue}.txt"
    if marker.exists():
        return "already sent"
    user, password, to = _account()
    if not (user and password and to):
        return "not configured"
    _deliver(compose(rows, names, site_url, user, to), user, password, cfg, smtp)
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text("sent\n", encoding="utf-8")
    return "sent"
