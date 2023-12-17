from matplotlib.pyplot import figure, savefig, show, tight_layout
from pandas import read_csv, DataFrame
from dslabs_functions import plot_horizontal_bar_chart

filename = "../../data/forecast_covid_single.csv"
file_tag = "Covid"
data: DataFrame = read_csv(
    "../../data/forecast_covid_single.csv",
    index_col="date",
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)

mv: dict[str, int] = {}
for var in data.columns:
    nr: int = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

# Ordenar o dicionário por valores (número de valores ausentes)
sorted_mv = dict(sorted(mv.items(), key=lambda item: item[1], reverse=True))

figure(figsize=(5, 7))
plot_horizontal_bar_chart(
    list(sorted_mv.keys()),
    list(sorted_mv.values()),
    title="Number of Missing Values per Variable",
    xlabel="No. of Missing Values",
    ylabel="Variables",
)
tight_layout()
savefig(f"images/{file_tag}_mv.png")
show()