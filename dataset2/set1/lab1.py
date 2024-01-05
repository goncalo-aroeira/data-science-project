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
    savefig(f"images/lab1/{file_tag}_records_variables.png")
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
    savefig(f"images/lab1/{file_tag}_variable_types.png")
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
    savefig(f"images/lab1/{file_tag}_mv.png")
    show()

# DATA SPARSITY
#***********************************************************************************
#*                                   EX 1                                          *
#*             4.1 - Scatter-plots (all x all - including class)                   *
#***********************************************************************************
import pandas as pd
from numpy import ndarray
from pandas import read_csv, DataFrame
from matplotlib.figure import Figure
from matplotlib.pyplot import figure, subplots, savefig, show
from dslabs_functions import HEIGHT, plot_multi_scatters_chart, plot_scatter_chart

def scatterPlots(data: DataFrame, file_tag: str, target: str):
    variable_types = get_variable_types(data)
    #data = data.head(10)
    out = ["Customer_ID","ID","SSN"]
    done = ["Month", "Name", "Age", "Occupation", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts" ,"Num_Credit_Card", "Interest_Rate"]
    data = data.dropna()

    vars: list = list(filter(lambda x: x not in out, data.columns.to_list()))
    print("all vars",vars)
    allPlots, allAxis = [], []
    
    if [] != vars:
        n: int = len(vars) - 1
        #fig, axs = subplots(n, n, figsize=(n * HEIGHT, n * HEIGHT), squeeze=False)
        for i in range(len(vars)):
            var1: str = vars[i]
            if var1 in done:
                continue
            print("var1 ", var1)
            rows = 1
            cols = n-i 
            print("rows " + str(rows) + "; cols " + str(cols))
            fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
            k = 0
            for j in range(i + 1, len(vars)):
                var2: str = vars[j]
                # binary X binary is not interesting
                if not (var1 in variable_types["binary"] and var2 in variable_types["binary"]):
                    #plot_multi_scatters_chart(data, var1, var2, target, ax=axs[i, j - 1])
                    plot_multi_scatters_chart(data, var1, var2, target, ax=axs[0, k])
                    k += 1
            savefig(f"images/lab1/{file_tag}_sparsity_{var1}_per_class.png", bbox_inches="tight")
        #show()
    else:
        print("Sparsity per class: there are no variables.")

#***********************************************************************************
#*                                   EX 2                                          *
#*              4.2 - Correlation (all x all - including class)                    *
#***********************************************************************************

from seaborn import heatmap
from dslabs_functions import get_variable_types

def correlationAll(data: DataFrame, file_tag: str):
    variables_types: dict[str, list] = get_variable_types(data)
    numeric: list[str] = variables_types["numeric"]
    corr_mtx: DataFrame = data[numeric].corr().abs()

    fig = figure(figsize=(8, 6))
    fig.suptitle(f"Correlation (all x all - including class)")
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
    savefig(f"images/lab1/{file_tag}_correlation_analysis.png", bbox_inches="tight")
    show()

#***********************************************************************************
#*                                   EX 2
#*                                Granularity                                      *
#*                          2.1 - Symbolic Variables                               *
#***********************************************************************************

from pandas import DataFrame, read_csv, Series
from matplotlib.pyplot import show, subplots, savefig
from matplotlib.figure import Figure
from numpy import ndarray, nan
from dslabs_functions import plot_bar_chart, HEIGHT, get_variable_types, define_grid

def analyse_property_granularity(data: DataFrame, axs: ndarray, j: int, vars: list[str]) -> ndarray:
    for i in range(len(vars)):
        counts: Series[int] = data[vars[i]].value_counts()
        if vars[i] == "Type_of_Loan":
                counts.index = list(map(abreviate_type_of_loan, counts.index.to_list()))
        plot_bar_chart(
            counts.index.to_list(),
            counts.to_list(),
            ax=axs[j, i],
            title=vars[i],
            xlabel=vars[i],
            ylabel="number of records",
            percentage=False,
        )
        if vars[i] == "Type_of_Loan" or vars[i] == "Credit_History_Age":
                axs[j, i].set_xticklabels([])
    if vars == ["Type_of_Loan", "Loan_grouped"]: # plotting the unique loans count on the same row
        plot_bar_chart(
            list(type_of_loan_unique_count(data).keys()),
            list(type_of_loan_unique_count(data).values()),
            ax=axs[j, i+1],
            title="Unique Loans Count",
            xlabel="Loan",
            ylabel="Number of Occurences",
            percentage=False,
        )
    return axs

def get_symbolic_nonBinary_variables(data: DataFrame) -> list:
    symbolic: list = get_variable_types(data)["symbolic"]
    remove_list: list = (["ID", "Customer_ID", "Name", "SSN", "Age"] + 
                         ["Month", "CreditMix", "Payment_of_Min_Amount"])
    res = []
    for el in symbolic:
        if el not in remove_list:
            res += [el]
    return res

def occupation_gran_data(occupation: str) -> str:
    STEM: list[str] = ["Scientist", "Engineer", "Developer", "Doctor", "Architect", "Mechanic"]
    Business: list[str] = ["Entrepreneur", "Manager"]
    Media: list[str] = ["Media_Manager", "Journalist"]
    Services: list[str] = ["Teacher", "Lawyer", "Accountant"]
    Creative: list[str] = ["Musician", "Writer"]
    if occupation in STEM:
        return "STEM"
    elif occupation in Business:
        return "Business"
    elif occupation in Media:
        return "Media"
    elif occupation in Services:
        return "Services"
    elif occupation in Creative:
        return "Creative"
    else:
        return nan
    
def credit_History_Age_gran_data(credit_history_age: str) -> str:
    if type(credit_history_age) == str:
        year = credit_history_age.split(" ")[:2]
        return " ".join(year)
    else:
        return nan

def pb_amount_spent_gran_data(pb: str) -> str:
    if type(pb) == str:
        amount_spent = pb.split("_")[:2]
        return "_".join(amount_spent)
    else:
        return nan
    
def pb_size_payments_gran_data(pb: str) -> str:
    if type(pb) == str:
        size_payments = pb.split("_")[2:]
        return "_".join(size_payments)
    else:
        return nan
    
def type_of_loan_gran_data(tl: str) -> str:
    if type(tl) == str:
        if "," in tl:
            return "multiple_loan"
        else:
            return "single_loan"
    else:
        return nan
    
def get_all_unique_loans(data: DataFrame) -> list:
    res = []
    for loan in data["Type_of_Loan"].dropna().unique().tolist():
        for individual_loan in list(map(lambda x: x.strip(), loan.split(','))):
            if "and " in individual_loan:
                individual_loan = individual_loan.replace("and ", "", 1)
            if individual_loan.replace(" ","_") not in res:
                res += [individual_loan.replace(" ","_")]
    return res

def get_unique_loans_for1entry(full_loan_entry: str) -> list:
    return list(map(lambda x: x.strip().replace("and ", "").replace(" ", "_"), full_loan_entry.split(',')))

def type_of_loan_unique_count(data: DataFrame):
    res = {}
    for unique_loan in get_all_unique_loans(data):
        res[unique_loan] = 0
    for full_loan_entry in data["Type_of_Loan"].dropna():
        for split_fle in get_unique_loans_for1entry(full_loan_entry):
            res[split_fle] += 1
    return res
            

def symbolic_variables_granularity(data: DataFrame, file_tag: str):
    variables = get_symbolic_nonBinary_variables(data)
    cols: int = 3
    rows: int = len(variables)
    fig: Figure
    axs: ndarray
    fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
    fig.suptitle(f"Granularity study for Occupation, Credit_History_Age, Payment_Behaviour and Type_of_Loan")
    varia: list[str]
    i: int = 0
    for var in variables:
        match var:
            case "Occupation":
                data["Area_of_Occupation"] = list(map(occupation_gran_data, data["Occupation"].values))
                varia = ["Occupation", "Area_of_Occupation"]
            case "Credit_History_Age":
                data["CHA_year"] = list(map(credit_History_Age_gran_data, data["Credit_History_Age"].values))
                varia = ["Credit_History_Age", "CHA_year"]
            case "Payment_Behaviour":
                data["PB_amount_spent"] = list(map(pb_amount_spent_gran_data, data["Payment_Behaviour"].values))
                data["PB_size_payments"] = list(map(pb_size_payments_gran_data, data["Payment_Behaviour"].values))
                varia = ["Payment_Behaviour", "PB_amount_spent", "PB_size_payments"]
            case "Type_of_Loan":
                data["Loan_grouped"] = list(map(type_of_loan_gran_data, data["Type_of_Loan"].values))
                varia = ["Type_of_Loan", "Loan_grouped"]
        analyse_property_granularity(data, axs, i, varia)
        i += 1
    # MAYBE CLEANUP ?????
    data.drop(["Area_of_Occupation", "CHA_year", "PB_amount_spent", "PB_size_payments", "Loan_grouped"], axis = 1)
    savefig(f"images/lab1/{file_tag}_granularity_v2.png", bbox_inches='tight')
    show()

#***********************************************************************************

#***********************************************************************************
#*                                   EX 3
#*                               Distribution                                      *
#***********************************************************************************

from dslabs_functions import set_chart_labels, plot_multiline_chart
from matplotlib.pyplot import xticks
from numpy import log
from scipy.stats import norm, expon, lognorm
from matplotlib.axes import Axes

#****************************  3.1-Global boxplot  *********************************

def global_box_plot(data: DataFrame, file_tag: str):
    variables_types: dict[str, list] = get_variable_types(data)
    numeric: list[str] = variables_types["numeric"]
    if [] != numeric:
        data.boxplot(rot=45)
        savefig(f"images/lab1/{file_tag}_global_boxplot.png", bbox_inches='tight')
        show()
    else:
        print("There are no numeric variables.")

#*******************  3.2-Boxplots for individual numeric vars  ********************

def boxplots_individual_num_vars(data: DataFrame, file_tag: str):
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
            i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
        savefig(f"images/lab1/{file_tag}_single_boxplots.png")
        show()
    else:
        print("There are no numeric variables.")

#***********************************************************************************

#********************************  3.3-Outliers  ***********************************

from dslabs_functions import plot_multibar_chart

NR_STDEV: int = 2
IQR_FACTOR: float = 1.5

def determine_outlier_thresholds_for_var(
    summary5: Series, std_based: bool = True, threshold: float = NR_STDEV
) -> tuple[float, float]:
    top: float = 0
    bottom: float = 0
    if std_based:
        std: float = threshold * summary5["std"]
        top = summary5["mean"] + std
        bottom = summary5["mean"] - std
    else:
        iqr: float = threshold * (summary5["75%"] - summary5["25%"])
        top = summary5["75%"] + iqr
        bottom = summary5["25%"] - iqr

    return top, bottom

def count_outliers(
    data: DataFrame,
    numeric: list[str],
    nrstdev: int = NR_STDEV,
    iqrfactor: float = IQR_FACTOR,
) -> dict:
    outliers_iqr: list = []
    outliers_stdev: list = []
    summary5: DataFrame = data[numeric].describe()

    for var in numeric:
        top: float
        bottom: float
        top, bottom = determine_outlier_thresholds_for_var(
            summary5[var], std_based=True, threshold=nrstdev
        )
        outliers_stdev += [
            data[data[var] > top].count()[var] + data[data[var] < bottom].count()[var]
        ]

        top, bottom = determine_outlier_thresholds_for_var(
            summary5[var], std_based=False, threshold=iqrfactor
        )
        outliers_iqr += [
            data[data[var] > top].count()[var] + data[data[var] < bottom].count()[var]
        ]

    return {"iqr": outliers_iqr, "stdev": outliers_stdev}

def outliers(data: DataFrame, file_tag: str):
    variables_types: dict[str, list] = get_variable_types(data)
    numeric: list[str] = variables_types["numeric"]

    if [] != numeric:
        outliers: dict[str, int] = count_outliers(data, numeric)
        figure(figsize=(20, HEIGHT))
        plot_multibar_chart(
            numeric,
            outliers,
            title="Nr of standard outliers per variable",
            xlabel="variables",
            ylabel="nr outliers",
            percentage=False,
        )
        savefig(f"images/lab1/{file_tag}_outliers_standard.png")
        show()
    else:
        print("There are no numeric variables.")

#***********************************************************************************

#**************************  3.4-Histograms for numeric  ***************************

def histograms_numeric_vars(data: DataFrame, file_tag: str):
    variables_types: dict[str, list] = get_variable_types(data)
    numeric: list[str] = variables_types["numeric"]

    if [] != numeric:
        rows: int
        cols: int
        rows, cols = define_grid(len(numeric))
        fig, axs = subplots(
            rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
        )
        i: int
        j: int
        i, j = 0, 0
        for n in range(len(numeric)):
            set_chart_labels(
                axs[i, j],
                title=f"Histogram for {numeric[n]}",
                xlabel=numeric[n],
                ylabel="nr records",
            )
            axs[i, j].hist(data[numeric[n]].dropna().values, 60)
            i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
        savefig(f"images/lab1/{file_tag}_single_histograms_numeric.png")
        show()
    else:
        print("There are no numeric variables.")

#***********************************************************************************

#************************  3.5-Distributions for numeric  **************************

def compute_known_distributions(x_values: list) -> dict:
    distributions = dict()
    # Gaussian
    mean, sigma = norm.fit(x_values)
    distributions["Normal(%.1f,%.2f)" % (mean, sigma)] = norm.pdf(x_values, mean, sigma)
    # Exponential
    loc, scale = expon.fit(x_values)
    distributions["Exp(%.2f)" % (1 / scale)] = expon.pdf(x_values, loc, scale)
    # LogNorm
    sigma, loc, scale = lognorm.fit(x_values)
    distributions["LogNor(%.1f,%.2f)" % (log(scale), sigma)] = lognorm.pdf(
        x_values, sigma, loc, scale
    )
    return distributions


def histogram_with_distributions(ax: Axes, series: Series, var: str):
    values: list = series.sort_values().to_list()
    ax.hist(values, 20, density=True)
    distributions: dict = compute_known_distributions(values)
    plot_multiline_chart(
        values,
        distributions,
        ax=ax,
        title="Best fit for %s" % var,
        xlabel=var,
        ylabel="",
    )


def distributions_numeric_vars(data: DataFrame, file_tag: str):
    variables_types: dict[str, list] = get_variable_types(data)
    numeric: list[str] = variables_types["numeric"]
    if [] != numeric:
        rows, cols = define_grid(len(numeric))
        fig, axs = subplots(
            rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False
        )
        i, j = 0, 0
        for n in range(len(numeric)):
            histogram_with_distributions(axs[i, j], data[numeric[n]].dropna(), numeric[n])
            i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
        savefig(f"images/lab1/{file_tag}_numeric_distribution.png")
        show()
    else:
        print("There are no numeric variables.")

#***********************************************************************************

#*************************  3.6-Histograms for symbolic  ***************************

def abreviate_type_of_loan(tl: str) -> str:
    res = ""
    for letter in tl:
        if letter.isupper() or letter == ",":
            res += letter
    return res

def histograms_symbolic_vars(data: DataFrame, file_tag: str):
    variables_types: dict[str, list] = get_variable_types(dataOg)
    symbolic: list[str] = variables_types["symbolic"] + variables_types["binary"]
    print(variables_types,symbolic)
    if [] != symbolic:
        rows, cols = define_grid(len(symbolic))
        fig, axs = subplots(
            rows, cols, figsize=(cols * HEIGHT*1.5, rows * HEIGHT*1.5), squeeze=False
        )
        i, j = 0, 0
        for n in range(len(symbolic)):
            counts: Series = data[symbolic[n]].value_counts()
            if symbolic[n] == "Type_of_Loan":
                counts.index = list(map(abreviate_type_of_loan, counts.index.to_list()))
            plot_bar_chart(
                counts.index.to_list(),
                counts.to_list(),
                ax=axs[i, j],
                title="Histogram for %s" % symbolic[n],
                xlabel=symbolic[n],
                ylabel="nr records",
                percentage=False,
            )
            
            i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
        savefig(f"images/lab1/{file_tag}_histograms_symbolic.png")
        show()
    else:
        print("There are no symbolic variables.")

#***********************************************************************************

#***************************  3.7-Class distribution  ******************************

def class_distribution(data: DataFrame, file_tag: str, target: str):
    values: Series = data[target].value_counts()
    figure(figsize=(4, 2))
    plot_bar_chart(
        values.index.to_list(),
        values.to_list(),
        title=f"Target distribution (target={target})",
    )
    savefig(f"images/lab1/{file_tag}_class_distribution.png", bbox_inches='tight')
    show()

#***********************************************************************************



if __name__ == "__main__":
    filenameOg = "data/class_credit_score.csv"
    filename = "data/ccs_vars_encoded.csv"
    file_tag = "Credit_Score"
    target = "Credit_Score"
    #data: DataFrame = read_csv(filename, na_values="", index_col="")
    data: DataFrame = read_csv(filename, na_values="", index_col="Unnamed: 0")
    dataOg: DataFrame = read_csv(filenameOg, na_values="")
    
    stroke: DataFrame = read_csv("data/stroke.csv", na_values="")

    #print(data.shape)
    #print(data.head)
    
    dataOg['Age'] = dataOg['Age'].astype(str).str.replace('_', '', regex=False).astype(int)
    
    # granularity
    symbolic_variables_granularity(dataOg, file_tag)

    # distribution
    #global_box_plot(data, file_tag)
    #boxplots_individual_num_vars(data, file_tag)
    #outliers(data, file_tag)
    #class_distribution(data, file_tag, target)
    #histograms_numeric_vars(data, file_tag)
    #distributions_numeric_vars(data, file_tag)
    #histograms_symbolic_vars(dataOg, file_tag)
    
    # Data Sparsity
    # scatterPlots(dataOg, file_tag, target)
    # correlationAll(data, file_tag)
