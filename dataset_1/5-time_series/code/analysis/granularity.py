from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show, savefig, subplots, close
from dslabs_functions import plot_line_chart, HEIGHT, ts_aggregation_by
from matplotlib.pyplot import subplots
from matplotlib.axes import Axes
from matplotlib.figure import Figure

file_tag = "Covid"
target = "deaths"
data: DataFrame = read_csv(
    "../../data/forecast_covid_single.csv",
    index_col="date",
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)
series: Series = data[target]


grans: list[str] = ["W", "M", "Q"]
fig: Figure
axs: list[Axes]
fig, axs = subplots(len(grans), 1, figsize=(3 * HEIGHT, 3* HEIGHT ))
fig.suptitle(f"{file_tag} {target} aggregation study")
funs: list[str] = ["sum", "mean"]

for fun in funs:
    fig: Figure
    axs: list[Axes]
    fig, axs = subplots(len(grans), 1, figsize=(3 * HEIGHT, 3* HEIGHT ))
    fig.suptitle(f"{file_tag} {target} aggregation study")
    for i in range(len(grans)):
        ss: Series = ts_aggregation_by(series, grans[i], agg_func=fun)
        plot_line_chart(
            ss.index.to_list(),
            ss.to_list(),
            ax=axs[i],
            xlabel=f"{ss.index.name} ({grans[i]})",
            ylabel=target,
            title=f"granularity={grans[i]}",
        )
    show()
    savefig(f"../images/{file_tag}_granularity_aggregation_{fun}.png")
    close()

