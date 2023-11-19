from matplotlib.pyplot import figure, savefig, show
from aux import plot_bar_chart
from pandas import read_csv, DataFrame

filename = "class_pos_covid.csv"
file_tag = "missing_values"
data: DataFrame = read_csv(filename)

mv: dict[str, int] = {}
for var in data.columns:
    nr: int = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

figure(figsize=(50, 50))
plot_bar_chart(
    list(mv.keys()),
    list(mv.values()),
    title="Nr of missing values per variable",
    xlabel="variables",
    ylabel="nr missing values",
    ytickfontsize=20,
)
savefig(f"images/{file_tag}_mv.png")
show()