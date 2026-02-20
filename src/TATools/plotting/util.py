from typing import Union, Tuple, List
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from tzlocal import get_localzone
FigSize = Tuple[float,float]
GraphReturn = Tuple[Figure, Union[Axes, List[Axes]]]

class CyclicList(list):
    def __init__(self, data):
        if isinstance(data, (str, bytes)):
            # treat strings and bytes as single items, not iterables
            data = [data]
        elif not hasattr(data, "__iter__"):
            # wrap non-iterables into a list
            data = [data]
        super().__init__(data)

    def __getitem__(self, i):
        if not self:
            raise IndexError("Cannot index into empty CyclicList")
        if isinstance(i, slice):
            start, stop, step = i.indices(len(self))
            rng = range(start, stop, step)
            return [super().__getitem__(j % len(self)) for j in rng]
        return super().__getitem__(i % len(self))

    def __iter__(self):
        if not self:
            return iter(())  # or raise, your call
        i = 0
        while True:
            yield super().__getitem__(i % len(self))
            i += 1
    
find_time_splits = lambda start_times, end_times: np.argwhere(start_times[1:] != end_times[:-1]).reshape(-1)+1
find_time_splits_period = lambda start_times, periods: find_time_splits(start_times, start_times+periods)


from typing import Iterable

timezone_now = lambda: pd.Timestamp.now().tz_localize(get_localzone()).tzname()
def decide_timezone(data: Iterable, localize_force: bool = False) -> str:
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        data = data.index
    if not all([isinstance(d, pd.Timestamp) for d in data]):
        # try to coerce to timestamp
        data = [pd.Timestamp(d) for d in data]
    if not all([d.tzinfo is not None for d in data]):
        # try to localize to system timezone
        data = [d.tz_localize(get_localzone()) for d in data]
    if localize_force:
        data = [d.tz_convert(get_localzone()) for d in data]
    timezones = np.unique([d.tzname() for d in data])
    if len(timezones) == 1:
        return timezones[0]
    elif len(timezones) == 2:
        # Handle common case of DST change, like "EST" and "EDT" -> "ET"
        if (timezones[0][0] == timezones[1][0]) and (timezones[0][-1] == timezones[1][-1]):
            return timezones[0][0] + timezones[0][-1] + " (mixed DST)"
    elif len(timezones) == 0:
        return ""
    return "Mixed Timezones"

def decide_timeformat(data: Union[pd.Series, pd.DataFrame, Iterable[Union[str, pd.Timestamp]]]) -> str:
    if isinstance(data, pd.Series) or isinstance(data, pd.DataFrame):
        data = data.index
    if not all([isinstance(d, pd.Timestamp) for d in data]):
        # try to coerce to timestamp
        data = [pd.Timestamp(d) for d in data]
    fmt = "%m-%d %H:%M"
    if not any([v.minute for v in data]):
        fmt = "%m-%d %H"
        if not any([v.hour for v in data]):
            fmt = "%m-%d"
    return fmt

def round_timedelta_reasonably(td: pd.Timedelta) -> pd.Timedelta:
    # First just round to minute
    td = td.round("1min")

    if td > pd.Timedelta("1h"):
        td = td.round("10min")
    if td > pd.Timedelta("6h"):
        td = td.round("1h")
    if td > pd.Timedelta("3 days"):
        td = td.round("6h")
    if td > pd.Timedelta("14 days"):
        td = td.round("24h")
    return td