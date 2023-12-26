from numpy import ndarray
from matplotlib.figure import Figure
from matplotlib.pyplot import savefig, show, subplots
from dslabs_functions import define_grid, HEIGHT, get_variable_types
from pandas import read_csv, DataFrame

filename = "../../class_pos_covid.csv"
file_tag = "CovidPos"
data: DataFrame = read_csv(filename)

variables_types: dict[str, list] = get_variable_types(data)
numeric: list[str] = variables_types["numeric"]

if [] != numeric:
    rows: int
    cols: int
    rows, cols = define_grid(len(numeric))
    fig: Figure
    axs: ndarray
    fig, axs = subplots(
        rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
    )
    i, j = 0, 0
    for n in range(len(numeric)):
        axs[i, j].set_title("Boxplot for %s" % numeric[n])
        axs[i, j].boxplot(data[numeric[n]].dropna().values)
        axs[i, j].set_xticklabels(["%s" % numeric[n]])
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)

    savefig(f"../images/{file_tag}_single_boxplots.png")
    show()
else:
    print("There are no numeric variables.")