from matplotlib.pyplot import subplots, show, savefig, close
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import read_csv, DataFrame, Series
from dslabs_functions import plot_line_chart, HEIGHT, ts_aggregation_by


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

ss_weeks: Series = ts_aggregation_by(series, gran_level="W", agg_func=sum)
ss_months: Series = ts_aggregation_by(series, gran_level="M", agg_func=sum)
ss_quarters: Series = ts_aggregation_by(series, gran_level="Q", agg_func=sum)

grans: list[Series] = [ss_weeks, ss_months, ss_quarters]
gran_names: list[str] = ["Weekly", "Monthly", "Quarterly"]

sizes: list[int] = [25, 50, 75, 100]
fig: Figure
axs: list[Axes]


for i in range(len(grans)):
    fig, axs = subplots(len(sizes), 1, figsize=(3 * HEIGHT, HEIGHT / 2 * len(sizes)))
    fig.suptitle(f"{file_tag} {target} {gran_names[i]} after smoothing")

    for j in range(len(sizes)):
        ss_smooth: Series = grans[i].rolling(window=sizes[j]).mean()
        plot_line_chart(
            ss_smooth.index.to_list(),
            ss_smooth.to_list(),
            ax=axs[j],
            xlabel=ss_smooth.index.name,
            ylabel=target,
            title=f"size={sizes[j]}",
        )
    show()
    savefig(f"images/{file_tag}_{target}_{gran_names[i]}_smoothing.png")
    close()
