from matplotlib.pyplot import figure, savefig, show
from aux import get_variable_types, plot_bar_chart
from pandas import read_csv, DataFrame

filename = "class_pos_covid.csv"
file_tag = "variable_types"
data: DataFrame = read_csv(filename)

variable_types: dict[str, list] = get_variable_types(data)
print(variable_types)
counts: dict[str, int] = {}
for tp in variable_types.keys():
    counts[tp] = len(variable_types[tp])

figure(figsize=(15, 15))
plot_bar_chart(
    list(counts.keys()), list(counts.values()), title="Nr of variables per type"
)
savefig(f"images/{file_tag}_variable_types.png")
show()