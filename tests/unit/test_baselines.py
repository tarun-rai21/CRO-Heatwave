"""Sliding training window wrapper."""

import pandas as pd

from heatwave.models.baselines import SlidingWindow


def test_sliding_window_trains_only_on_the_last_years():
    class Spy:
        name = "spy"

        def fit(self, train):
            self.seen = train
            return self

    dates = pd.to_datetime([f"{y}-05-01" for y in range(1981, 2006)])
    train = pd.DataFrame({"valid_date": dates})
    m = SlidingWindow(Spy(), "lead", 20).fit(train)
    assert m.model.seen["valid_date"].dt.year.tolist() == list(range(1986, 2006))
