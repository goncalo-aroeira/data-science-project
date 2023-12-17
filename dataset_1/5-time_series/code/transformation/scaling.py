from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show, savefig, subplots, tight_layout
from dslabs_functions import plot_line_chart, HEIGHT, scale_all_dataframe, ts_aggregation_by
from sklearn.model_selection import train_test_split


    
file_tag = "Covid"
filename = "../../data/forecast_covid_single.csv"
index = "date"
target = "deaths"

data: DataFrame = read_csv(
    filename,
    index_col=index,
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)
series: Series = data[target]

df: DataFrame = scale_all_dataframe(data)
ss: Series = df[target]

ss_weeks: Series = ts_aggregation_by(ss, gran_level="W", agg_func=sum)
ss_months: Series = ts_aggregation_by(ss, gran_level="M", agg_func=sum)
ss_quarters: Series = ts_aggregation_by(ss, gran_level="Q", agg_func=sum)

grans: list[Series] = [ss_weeks, ss_months, ss_quarters]
gran_names: list[str] = ["Weekly", "Monthly", "Quarterly"]

fig, axs = subplots(3, 1, figsize=(10, 15))  # Create 3 subplots vertically

for i in range(len(grans)):
    axs[i].plot(
        grans[i].index.to_list(),
        grans[i].to_list(),
    )
    axs[i].set_xlabel(grans[i].index.name)
    axs[i].set_ylabel(target)
    axs[i].set_title(f"{file_tag} {target} {gran_names[i]}")

tight_layout()  
savefig(f"images/{file_tag}_{target}_scaling.png")  
show()
