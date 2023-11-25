from matplotlib.pyplot import figure, savefig, show, tight_layout
from pandas import read_csv, DataFrame
from dataset_1.dslabs_functions import plot_horizontal_bar_chart

filename = "../../class_pos_covid.csv"
file_tag = "CovidPos"
data: DataFrame = read_csv(filename)

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
savefig(f"../images/{file_tag}_mv.png")
show()
