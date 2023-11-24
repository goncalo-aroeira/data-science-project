import matplotlib.pyplot as plt
from pandas import DataFrame
from dataset_1.dslabs_functions import get_variable_types
from pandas import read_csv

filename = "../../class_pos_covid.csv"
file_tag = "CovidPos"
data: DataFrame = read_csv(filename)

variables_types: dict[str, list] = get_variable_types(data)
numeric: list[str] = variables_types["numeric"]
corr_mtx: DataFrame = data[numeric].corr().abs()

# Create a heatmap using matplotlib
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(abs(corr_mtx), cmap='Blues')
plt.xticks(range(len(numeric)), numeric, rotation=45, ha='left')
plt.yticks(range(len(numeric)), numeric)
plt.colorbar(cax)

plt.savefig(f"../images/{file_tag}_correlation_analysis.png")
plt.show()