"""IMD gridded Tmax: download validation, binary layout, resumable ingest (no network)."""

from datetime import date

import numpy as np
import pytest
import requests

from heatwave.config import read_yaml
from heatwave.ingest import imd
from heatwave.ingest.imd import ImdError, ImdSettings, download_year, read_grd, to_long

S = ImdSettings.from_config(read_yaml("data.yaml")["imd"])


def year_bytes(year, fill=30.0, set_cell=None):
    """A synthetic IMD file: every land value = fill; sea (first row) = 99.9.
    set_cell = (day, lat, lon, value) overrides one value."""
    days = 366 if year % 4 == 0 else 365
    a = np.full((days, S.n_lat, S.n_lon), fill, dtype="<f4")
    a[:, 0, :] = S.missing_value
    if set_cell:
        d, la, lo, v = set_cell
        a[
            d,
            int(round((la - S.lat_first) / S.step_deg)),
            int(round((lo - S.lon_first) / S.step_deg)),
        ] = v
    return a.tobytes()


class FakeResp:
    def __init__(self, status, content):
        self.status_code, self.content = status, content


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def post(self, url, data, headers, timeout):
        self.calls.append(data)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def test_expected_size():
    assert S.expected_bytes(2024) == 366 * 31 * 31 * 4  # 1,406,904: matches a real download
    assert S.expected_bytes(2023) == 365 * 31 * 31 * 4


def test_html_error_page_with_status_200_is_retried():
    good = year_bytes(2023)
    session = FakeSession(
        [
            requests.ConnectionError("reset"),
            FakeResp(200, b"<html>error</html>"),
            FakeResp(200, good),
        ]
    )
    waits = []
    assert download_year(S, 2023, session=session, sleep=waits.append) == good
    assert len(waits) == 2
    assert session.calls[0] == {S.form_field: 2023}


def test_gives_up_with_reason():
    session = FakeSession([FakeResp(200, b"not published")] * (S.max_retries + 1))
    with pytest.raises(ImdError, match="expected"):
        download_year(S, 2023, session=session, sleep=lambda s: None)


def test_layout_is_days_lat_lon_ascending(tmp_path):
    # Put 45.0 at day 140, 25.5 N, 80.5 E and check it lands there and only there.
    path = tmp_path / "2024.GRD"
    path.write_bytes(year_bytes(2024, set_cell=(140, 25.5, 80.5, 45.0)))
    data = read_grd(path, 2024, S)
    assert data.shape == (366, 31, 31)
    j, i = list(S.lats).index(25.5), list(S.lons).index(80.5)
    assert data[140, j, i] == 45.0
    assert np.nansum(data == 45.0) == 1


def test_missing_value_becomes_nan(tmp_path):
    path = tmp_path / "2023.GRD"
    path.write_bytes(year_bytes(2023))
    data = read_grd(path, 2023, S)
    assert np.isnan(data[:, 0, :]).all()
    assert not np.isnan(data[:, 1:, :]).any()


def test_wrong_size_file_rejected(tmp_path):
    path = tmp_path / "2023.GRD"
    path.write_bytes(year_bytes(2024))  # leap-year file under a non-leap year
    with pytest.raises(ImdError):
        read_grd(path, 2023, S)


def test_to_long_drops_sea_and_applies_bbox(tmp_path):
    path = tmp_path / "2024.GRD"
    path.write_bytes(year_bytes(2024, set_cell=(140, 25.5, 80.5, 45.0)))
    long = to_long(read_grd(path, 2024, S), 2024, S, bbox=(77.0, 23.5, 84.5, 30.5))
    assert long["lat"].between(23.5, 30.5).all() and long["lon"].between(77.0, 84.5).all()
    assert not long["tmax"].isna().any()
    hit = long[long["tmax"] == 45.0]
    assert hit[["lat", "lon"]].values.tolist() == [[25.5, 80.5]]
    assert hit["obs_date"].iloc[0] == date(2024, 5, 20)  # day index 140 in a leap year


def test_ingest_years_is_resumable(tmp_path, monkeypatch):
    calls = []

    def fake_download(s, year, sleep=None):
        calls.append(year)
        if year == 2022:
            raise ImdError("server down")
        return year_bytes(year)

    monkeypatch.setattr(imd, "download_year", fake_download)
    with pytest.raises(ImdError, match="2022"):
        imd.ingest_years(S, tmp_path, [2021, 2022, 2023], sleep=lambda s: None)
    assert calls == [2021, 2022, 2023]

    calls.clear()
    monkeypatch.setattr(
        imd, "download_year", lambda s, year, sleep=None: (calls.append(year), year_bytes(year))[1]
    )
    manifest = imd.ingest_years(S, tmp_path, [2021, 2022, 2023], sleep=lambda s: None)
    assert calls == [2022]  # only the missing year
    assert manifest["year"].tolist() == [2021, 2022, 2023]
    assert manifest["bytes"].tolist() == [S.expected_bytes(y) for y in (2021, 2022, 2023)]
