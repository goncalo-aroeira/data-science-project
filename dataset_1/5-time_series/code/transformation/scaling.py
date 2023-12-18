from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show, savefig, subplots, tight_layout
from dslabs_functions import plot_line_chart, HEIGHT, scale_all_dataframe, ts_aggregation_by
from sklearn.model_selection import train_test_split


    
file_tag = "Covid"
filename = "../../data/forecast_Covid_first_derivative.csv"
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
df.to_csv(f"../../data/forecast_{file_tag}_scaled.csv")
ss: Series = df[target]


df: DataFrame = scale_all_dataframe(data)

ss: Series = df[target]
figure(figsize=(3 * HEIGHT, HEIGHT / 2))
plot_line_chart(
    ss.index.to_list(),
    ss.to_list(),
    xlabel=ss.index.name,
    ylabel=target,
    title=f"{file_tag} {target} after scaling",
)
savefig(f"images/{file_tag}_{target}_scaling.png")  
show()
