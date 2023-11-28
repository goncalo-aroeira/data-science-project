from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, savefig, close
from dslabs_functions import plot_bar_chart, mvi_by_filling, get_variable_types, evaluate_approach, plot_multibar_chart, mvi_by_dropping


filename = "../../../class_pos_covid.csv"
file_tag = "CovidPos"
target = "CovidPos"

data: DataFrame = read_csv(filename, na_values="")
data_copy = data.copy()
print(f"Dataset nr records={data.shape[0]}", f"nr variables={data.shape[1]}")

mv: dict[str, int] = {}
figure(figsize=(28, 6))
for var in data:
    nr: int = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

plot_bar_chart(
    list(mv.keys()),
    list(mv.values()),
    title="Missing values per variable",
    xlabel="variables",
    ylabel="nr missing values",
)
savefig(f"../images/{file_tag}_missing_values_per_variable.png")


############################################# MV Imputation #############################################

vars_with_mv: list = []
for var in data.columns:
    if data[var].isna().sum() > 0:
        vars_with_mv += [var]
print(f"variables with missing values: {vars_with_mv}")

# no variable has a considerable amount of missing values therefore we wont drop columns
# remove rows with a lot of missing values (80%) - number of columns = 40
MIN_MV_IN_A_RECORD_RATIO = 0.8
data.dropna(axis=0, thresh=round(data.shape[1] * MIN_MV_IN_A_RECORD_RATIO, 0), inplace=True)

og_symb_vars = get_variable_types(data)["symbolic"]
og_num_vars = get_variable_types(data)["numeric"]

#data_filling_frequent = mvi_by_filling(data, "frequent", og_symb_vars, og_num_vars)
data_filling_frequent = mvi_by_dropping(data, og_symb_vars, og_num_vars)
data_filling_frequent.to_csv("../data/CovidPos_mvi_fill_frequent.csv")
#data_filling_knn = mvi_by_filling(data_copy, "knn", og_symb_vars, og_num_vars, 3)
data_filling_knn = mvi_by_dropping(data_copy, og_symb_vars, og_num_vars)
data_filling_knn.to_csv("../data/CovidPos_mvi_fill_knn.csv")


############################################# MV Evaluation #############################################
frequent_fn = "../data/CovidPos_mvi_fill_frequent.csv"
knn_fn = "../data/CovidPos_mvi_fill_knn.csv"
data_frequent_mvi_fill: DataFrame = read_csv(frequent_fn)
data_knn_mvi_fill: DataFrame = read_csv(knn_fn)

figure()
eval: dict[str, list] = evaluate_approach(data_frequent_mvi_fill.head(int(data_frequent_mvi_fill.shape[0]*0.8)), 
                                            data_frequent_mvi_fill.tail(int(data_frequent_mvi_fill.shape[0]*0.2)), 
                                            target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"{file_tag} evaluation", percentage=True
)
savefig(f"../images/{file_tag}_mvi_freq_eval.png")
close()

figure()
eval: dict[str, list] = evaluate_approach(data_knn_mvi_fill.head(int(data_knn_mvi_fill.shape[0]*0.8)), 
                                            data_knn_mvi_fill.tail(int(data_knn_mvi_fill.shape[0]*0.2)), 
                                            target=target, metric="recall")
plot_multibar_chart(
    ["NB", "KNN"], eval, title=f"{file_tag} evaluation", percentage=True
)
savefig(f"../images/{file_tag}_mvi_knn_eval.png")
