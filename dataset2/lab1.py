from pandas import read_csv, DataFrame

filename = "data/class_credit_score.csv"
file_tag = "Credit_Score"
data: DataFrame = read_csv(filename, na_values="", index_col="ID")

print(data.shape)
print(data.head)
data['Age'] = data['Age'].astype(str).str.replace('_', '', regex=False)

#***********************************************************************************
#*                                   EX 1                                          *
#*                         Nr records x Nr variables                               *
#***********************************************************************************
from matplotlib.pyplot import figure, savefig, show, tight_layout
from dslabs_functions import plot_bar_chart

# figure(figsize=(4, 2))
# values: dict[str, int] = {"nr records": data.shape[0], "nr variables": data.shape[1]}
# plot_bar_chart(
#     list(values.keys()), list(values.values()), title="Nr of records vs nr variables"
# )
# savefig(f"images/{file_tag}_records_variables.png")
# show()

#***********************************************************************************
#*                                   EX 2                                          *
#*                               Variable types                                    *
#***********************************************************************************

from pandas import Series, to_numeric, to_datetime


def get_variable_types(df: DataFrame) -> dict[str, list]:
    variable_types: dict = {"numeric": [], "binary": [], "date": [], "symbolic": []}

    nr_values: Series = df.nunique(axis=0, dropna=True)
    for c in df.columns:
        if 2 == nr_values[c]:
            variable_types["binary"].append(c)
            df[c].astype("bool")
        else:
            try:
                to_numeric(df[c], errors="raise")
                variable_types["numeric"].append(c)
            except ValueError:
                try:
                    df[c] = to_datetime(df[c], errors="raise")
                    variable_types["date"].append(c)
                except ValueError:
                    variable_types["symbolic"].append(c)

    return variable_types
variable_types: dict[str, list] = get_variable_types(data)

print(variable_types)
counts: dict[str, int] = {}
for tp in variable_types.keys():
    counts[tp] = len(variable_types[tp])

figure(figsize=(4, 2))
plot_bar_chart(
    list(counts.keys()), list(counts.values()), title="Nr of variables per type"
)
savefig(f"images/{file_tag}_variable_types.png")
show()


#***********************************************************************************
#*                                   EX 3                                          *
#*                             Nr missing values                                   *
#***********************************************************************************
from dslabs_functions import plot_horizontal_bar_chart

# mv: dict[str, int] = {}
# for var in data.columns:
#     nr: int = data[var].isna().sum()
#     if nr > 0:
#         mv[var] = nr

# # print(mv)
# figure(figsize=(5, 3))
# plot_horizontal_bar_chart(
#     list(mv.keys()),
#     list(mv.values()),
#     title="Nr of missing values per variable",
#     xlabel="nr missing values",
#     ylabel="variables",
# )
# tight_layout()
# savefig(f"images/{file_tag}_mv.png")
# show()

# DATA SPARSITY
#***********************************************************************************
#*                                   EX 1                                          *
#*                Scatter-plots (all x all - including class)                      *
#***********************************************************************************
# import pandas as pd
# from numpy import ndarray
# from pandas import read_csv, DataFrame
# from matplotlib.figure import Figure
# from matplotlib.pyplot import figure, subplots, savefig, show
# from dslabs_functions import HEIGHT, plot_multi_scatters_chart

# data = data.dropna()
# vars = pd.DataFrame()
# for col in data.select_dtypes(include="number").columns:
#     vars[col] = pd.to_numeric(data[col], errors='coerce')
# symbolic_vars = data.select_dtypes(include="object").columns
# vars_symbolic = pd.DataFrame()

# for col in symbolic_vars:
#     vars_symbolic[col], _ = pd.factorize(data[col])

# # Combine numeric and factorized symbolic variables
# vars_combined = pd.concat([vars, vars_symbolic], axis=1)

# n = len(vars_combined.columns)
# fig, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)

# for i in range(len(vars_combined.columns)):
#     var1 = vars_combined.columns[i]
#     for j in range(i + 1, len(vars_combined.columns)):
#         var2 = vars_combined.columns[j]
#         plot_multi_scatters_chart(vars_combined, var1, var2, ax=axs[i, j - 1])

#***********************************************************************************
#*                                   EX 2                                          *
#*                 Correlation (all x all - including class)                       *
#***********************************************************************************

from seaborn import heatmap
from dslabs_functions import get_variable_types

# variables_types: dict[str, list] = get_variable_types(data)
# numeric: list[str] = variables_types["numeric"]
# corr_mtx: DataFrame = data[numeric].corr().abs()

# fig = figure(figsize=(5, 4))
# fig.suptitle(f"Scatter-plots (all x all - including class)")
# ax = heatmap(
#     abs(corr_mtx),
#     xticklabels=numeric,
#     yticklabels=numeric,
#     annot=False,
#     cmap="Blues",
#     vmin=0,
#     vmax=1,
# )
# ax.set_xticklabels(numeric, rotation=40, ha='right')
# tight_layout()
# savefig(f"images/{file_tag}_correlation_analysis.png")
# show()