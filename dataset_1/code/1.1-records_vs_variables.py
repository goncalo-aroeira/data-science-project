from matplotlib.pyplot import figure, savefig, show
from dslabs_functions import plot_bar_chart
from pandas import read_csv, DataFrame

filename = "../../class_pos_covid.csv"
file_tag = "CovidPos"
data: DataFrame = read_csv(filename)

figure(figsize=(5, 5))
values: dict[str, int] = {"nr records": data.shape[0], "nr variables": data.shape[1]}
plot_bar_chart(
    list(values.keys()), list(values.values()), title="Number of Records vs Number of Variables"
)
savefig(f"../images/{file_tag}_records_variables.png")
show()