from matplotlib.pyplot import figure, savefig, show
from pandas import read_csv, DataFrame, Series
from dslabs_functions import plot_bar_chart

filename = "class_pos_covid.csv"
file_tag = "CovidPos"
data: DataFrame = read_csv(filename)
target = "CovidPos"

values: Series = data[target].value_counts()
print(values)

figure(figsize=(4, 2))
plot_bar_chart(
    values.index.to_list(),
    values.to_list(),
    title=f"Target distribution (target={target})",
)
savefig(f"images/{file_tag}_class_distribution.png")
show()