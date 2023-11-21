from pandas import read_csv, DataFrame

#***********************************************************************************
#*                                   EX 1                                          *
#*                         Nr records x Nr variables                               *
#***********************************************************************************
from matplotlib.pyplot import figure, savefig, show, tight_layout
from dslabs_functions import plot_bar_chart

def nrRecordsVars(data: DataFrame, file_tag: str):
    figure(figsize=(4, 2))
    values: dict[str, int] = {"nr records": data.shape[0], "nr variables": data.shape[1]}
    plot_bar_chart(
        list(values.keys()), list(values.values()), title="Nr of records vs nr variables"
    )
    savefig(f"images/{file_tag}_records_variables.png")
    show()

#***********************************************************************************
#*                                   EX 2                                          *
#*                               Variable types                                    *
#***********************************************************************************

from pandas import Series, to_numeric, to_datetime

def variableTypes(data: DataFrame, file_tag: str):
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

def missingVals(data: DataFrame, file_tag: str):
    mv: dict[str, int] = {}
    for var in data.columns:
        nr: int = data[var].isna().sum()
        print(var + '-' + str(nr))
        if nr > 0:
            mv[var] = nr
    sorted_mv = dict(sorted(mv.items(), key=lambda item: item[1], reverse=True))
    print(mv)
    print(sorted_mv)
    print(data.columns)
    figure(figsize=(5, 3))
    plot_horizontal_bar_chart(
        list(sorted_mv.keys()),
        list(sorted_mv.values()),
        title="Nr of missing values per variable",
        xlabel="nr missing values",
        ylabel="variables",
    )

    tight_layout()
    savefig(f"images/{file_tag}_mv.png")
    show()

# DATA SPARSITY
#***********************************************************************************
#*                                   EX 1                                          *
#*                Scatter-plots (all x all - including class)                      *
#***********************************************************************************
import pandas as pd
from numpy import ndarray
from pandas import read_csv, DataFrame
from matplotlib.figure import Figure
from matplotlib.pyplot import figure, subplots, savefig, show
from dslabs_functions import HEIGHT, plot_multi_scatters_chart, plot_scatter_chart

def scatterPlots(data: DataFrame, file_tag: str):
    data = data.dropna()

    vars: list = data.columns.to_list()
    print("all vars",vars)
    allPlots, allAxis = [], []
    
    if [] != vars:
        target = "Credit_Score"
        out = ["Customer_ID","ID","SSN"]
        n: int = len(vars) - 1
        fig, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
        for i in range(len(vars)):
            var1: str = vars[i]
            if var1 in out:
                continue
            for j in range(i + 1, len(vars)):
                var2: str = vars[j]
                if var2 in out:
                    continue
                allPlots += [[var1,var2]]
                allAxis += [[i, j - 1]]
                # plot_multi_scatters_chart(data, var1, var2, target, ax=axs[i, j - 1])
        # savefig(f"images/{file_tag}_sparsity_per_class_study.png")
        # show()
        print(allPlots, len(allPlots))
        for i in range(len(allPlots)):
            var1, var2 = allPlots[i][0], allPlots[i][1]
            plot_multi_scatters_chart(data, var1, var2, target, ax=axs[allAxis[i][0], allAxis[i][1]])
        savefig(f"images/{file_tag}_ll_sparsity_per_class_study.png")
        show()
        # for i in range(3,39):
        #     var1, var2 = allPlots[i][0], allPlots[i][1]
        #     plot_multi_scatters_chart(data, var1, var2, target, ax=axs[allAxis[i][0], allAxis[i][1]])
        # print("loop01 done")
        # savefig(f"images/{file_tag}01_sparsity_per_class_study.png")
        # print("loop1 start")
        
        # for i in range(39,117):
        #     var1, var2 = allPlots[i][0], allPlots[i][1]
        #     plot_multi_scatters_chart(data, var1, var2, target, ax=axs[allAxis[i][0], allAxis[i][1]])
        # print("loop1 done")
        
        # savefig(f"images/{file_tag}1_sparsity_per_class_study.png")
        # print("starting loop2")
        # for i in range(117, 234):
        #     var1, var2 = allPlots[i][0], allPlots[i][1]
        #     plot_multi_scatters_chart(data, var1, var2, target, ax=axs[allAxis[i][0], allAxis[i][1]])
        # print("loop2 done")
        # savefig(f"images/{file_tag}2_sparsity_per_class_study.png")
        # print("starting loop3")
        # for i in range(234, 351):
        #     var1, var2 = allPlots[i][0], allPlots[i][1]
        #     plot_multi_scatters_chart(data, var1, var2, target, ax=axs[allAxis[i][0], allAxis[i][1]])
        # print("loop3 done")
        # savefig(f"images/{file_tag}3_sparsity_per_class_study.png")
        # print("loop3 saved")
    
    else:
        print("Sparsity per class: there are no variables.")
    
    print("Sparsity done")

#***********************************************************************************
#*                                   EX 2                                          *
#*                 Correlation (all x all - including class)                       *
#***********************************************************************************

from seaborn import heatmap
from dslabs_functions import get_variable_types

def correlationAll(data: DataFrame, file_tag: str):
    variables_types: dict[str, list] = get_variable_types(data)
    numeric: list[str] = variables_types["numeric"]
    corr_mtx: DataFrame = data[numeric].corr().abs()

    fig = figure(figsize=(5, 4))
    fig.suptitle(f"Scatter-plots (all x all - including class)")
    ax = heatmap(
        abs(corr_mtx),
        xticklabels=numeric,
        yticklabels=numeric,
        annot=False,
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    ax.set_xticklabels(numeric, rotation=40, ha='right')
    tight_layout()
    savefig(f"images/{file_tag}_correlation_analysis.png")
    show()

#***********************************************************************************
#*                                   EX 2
#*                                Granularity                                      *
#*                           4 - Symbolic Variables                                *
#***********************************************************************************

from pandas import DataFrame, read_csv, Series
from matplotlib.pyplot import show, subplots, savefig
from matplotlib.figure import Figure
from numpy import ndarray
from dslabs_functions import plot_bar_chart, HEIGHT, get_variable_types, define_grid

def analyse_property_granularity(data: DataFrame, property: str, vars: list[str]) -> ndarray:
    rows: int
    cols: int
    rows, cols = define_grid(len(vars))
    fig: Figure
    axs: ndarray
    fig, axs = subplots(1, cols, figsize=(cols * HEIGHT, HEIGHT), squeeze=False)
    fig.suptitle(f"Granularity study for {property}")
    for i in range(cols):
        counts: Series[int] = data[vars[i]].value_counts()
        plot_bar_chart(
            counts.index.to_list(),
            counts.to_list(),
            ax=axs[0, i],
            title=vars[i],
            xlabel=vars[i],
            ylabel="number of records",
            percentage=False,
        )
    return axs


def get_symbolic_nonBinary_variables(data: DataFrame) -> list:
    symbolic: list = get_variable_types(data)["symbolic"]
    remove_list: list = ["ID", "Customer_ID", "Name", "SSN", "Age"]  
    res = []
    for el in symbolic:
        if el not in remove_list:
            res += [el]
    return res


def symbolic_variables_granularity(data: DataFrame, file_tag: str):
    analyse_property_granularity(data, "Symbolic Variables", get_symbolic_nonBinary_variables(data))
    savefig(f"images/{file_tag}_granularity.png")
    show()

#***********************************************************************************


if __name__ == "__main__":
    filename = "data/class_credit_score.csv"
    file_tag = "Credit_Score"
    data: DataFrame = read_csv(filename, na_values="", index_col="ID")

    print(data.shape)
    print(data.head)
    data['Age'] = data['Age'].astype(str).str.replace('_', '', regex=False)

    # granularity
    scatterPlots(data, file_tag)
