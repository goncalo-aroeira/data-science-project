from matplotlib.pyplot import savefig, show, subplots
from dslabs_functions import get_variable_types
from pandas import read_csv, DataFrame

filename = "../../class_pos_covid.csv"
file_tag = "CovidPos"
data: DataFrame = read_csv(filename)

variables_types: dict[str, list] = get_variable_types(data)
numeric: list[str] = variables_types["numeric"]
if [] != numeric:
    fig, ax = subplots(figsize=(10, 8))
    data[numeric].boxplot(rot=45)
    savefig(f"../images/{file_tag}_global_boxplot.png")
    show()
else:
    print("There are no numeric variables.")