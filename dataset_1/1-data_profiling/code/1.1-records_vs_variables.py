from matplotlib.pyplot import figure, savefig, show
from dataset_1.dslabs_functions import plot_bar_chart
from pandas import read_csv, DataFrame

filename = "../../class_pos_covid.csv"
file_tag = "CovidPos"
data: DataFrame = read_csv(filename)

figure(figsize=(5, 5))
values: dict[str, int] = {"No. of Records": data.shape[0], "No. of Variables": data.shape[1]}
plot_bar_chart(
    list(values.keys()), list(values.values()), title="Comparison between Number of Records and Variables"
)
savefig(f"../images/{file_tag}_records_variables.png")
show()