import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union
from pandas import Timedelta
from tzlocal import get_localzone
from .util import Axes, Figure, round_timedelta_reasonably, decide_timezone, decide_timeformat

def plot_duet_sample_rate_bar(df: pd.DataFrame, resample_period: Optional[Union[str, Timedelta]] = None, measurement_period_duet: str = "5s",
                              figsize = (16, 5), plot_target_rate_line: bool = True, timestamps_as_insertion_time: bool = False, verbose: bool = False) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(1,1, figsize=figsize)

    if resample_period is None:
        resample_period = max([
            pd.Timedelta("1min"),
            (lambda arr: arr[-1] - arr[0])(df.index.sort_values()) / 8
        ])
        resample_period = round_timedelta_reasonably(resample_period)
        if verbose:
            print(f"Auto-selected resample_period: {resample_period}")
    if plot_target_rate_line:
        ax.axhline(pd.Timedelta(resample_period) / pd.Timedelta(measurement_period_duet), color="black", linestyle="dashed", alpha=.5, label="Target")

    d = df.copy()
    d['@timestamp'] = pd.to_datetime(d['@timestamp'].values).tz_convert(get_localzone())
    if timestamps_as_insertion_time:
        d.set_index("@timestamp", inplace=True)

    sns.barplot(
        d.groupby("serial_number").resample(rule=resample_period)[['temp']].count()[['temp']].rename(columns={"temp": "count"}),
        x=d.index.name,
        y="count",
        hue="serial_number",
        ax=ax
    )
    ax.set_title("Data Rate")

    ticklabel_values = [pd.Timestamp(t.get_text()) for t in ax.get_xticklabels()]
    fmt = decide_timeformat(ticklabel_values)
    ax.set_xticklabels([v.strftime(fmt) for v in ticklabel_values], rotation=90)
    ax.set_xlabel(decide_timezone(df.index))

    # set_datetime_xaxis_format(ax, date_format="%m-%d %H:%M", data_index=d.index, rotation=90, label_tz=True)

    fig.tight_layout(rect=(0, 0, 1, .8))
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig, ax