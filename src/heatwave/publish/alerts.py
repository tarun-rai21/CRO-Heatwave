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
    user, password, to = (
        os.environ.get("ALERT_SMTP_USER"),
        os.environ.get("ALERT_SMTP_PASSWORD"),
        os.environ.get("ALERT_TO"),
    )
    if not (user and password and to):
        return "not configured"
    msg = compose(rows, names, site_url, user, to)
    connect = smtp or (lambda: smtplib.SMTP_SSL(cfg["smtp_host"], cfg["smtp_port"], timeout=60))
    with connect() as server:
        server.login(user, password)
        server.send_message(msg)
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text("sent\n", encoding="utf-8")
    return "sent"
