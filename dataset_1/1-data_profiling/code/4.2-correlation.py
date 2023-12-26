import matplotlib.pyplot as plt
from pandas import DataFrame
from dslabs_functions import get_variable_types
from pandas import read_csv

filename = "../../2-data_preparation/data/ccs_vars_encoded.csv"
file_tag = "CovidPos"
data: DataFrame = read_csv(filename)

variables_types: dict[str, list] = get_variable_types(data)
numeric: list[str] = variables_types["numeric"]
binary: list[str] = variables_types["binary"]
corr_mtx: DataFrame = data[numeric+binary].corr().abs()

# Create a heatmap using matplotlib
fig, ax = plt.subplots(figsize=(10, 8))
cax = ax.matshow(abs(corr_mtx), cmap='Blues')
plt.xticks(range(len(numeric)+len(binary)), numeric+binary, rotation=45, ha='left')
plt.yticks(range(len(numeric)+len(binary)), numeric+binary)
plt.colorbar(cax)

plt.savefig(f"../images/{file_tag}_correlation_analysis.png")
plt.show()