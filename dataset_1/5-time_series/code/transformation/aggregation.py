from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show, savefig, subplots, tight_layout
from dslabs_functions import plot_line_chart, ts_aggregation_by, HEIGHT, plot_ts_multivariate_chart

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

fig, axs = subplots(3, 1, figsize=(10, 15))  

for i in range(len(grans)):
    plot_line_chart(
        grans[i].index.to_list(),
        grans[i].to_list(),
        xlabel=grans[i].index.name,
        ylabel=target,
        title=f"{file_tag} {target} {gran_names[i]}",
        ax=axs[i]  
    )

tight_layout()  
savefig(f"images/{file_tag}_{target}_aggregation.png")  
show() 
