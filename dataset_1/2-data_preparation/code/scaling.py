from pandas import read_csv, DataFrame, Series
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.pyplot import figure, savefig, close, show, subplots
from dslabs_functions import  plot_multibar_chart, evaluate_approach
from sklearn.model_selection import train_test_split


def scaling_evaluation(data_filename: str, strategy: str, target: str):
    data: DataFrame = read_csv(data_filename)

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    ## Evaluate Approach
    file_tag = "CovidPos"
    target = "CovidPos"

    figure()
    eval: dict[str, list] = evaluate_approach(train_data, test_data, target=target, metric="recall")
    plot_multibar_chart(
        ["NB", "KNN"], eval, title=f"{file_tag} evaluation", percentage=True)

    savefig(f"../images/CovidPos_scaling_treat_{strategy}.png")
    show()

data_filename: str = "../data/CovidPos_outliers_trunc_minmax.csv"
file_tag: str = "CovidPos"
target = "CovidPos"

data: DataFrame = read_csv(data_filename)
#scaling_evaluation(data_filename, "Original", target)

# Approach 1 - Standard scaling (z-score transformation)
vars: list[str] = data.columns.to_list()
vars.remove(target)
target_data: Series = data.pop(target)

transf: StandardScaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(data)
df_zscore = DataFrame(transf.transform(data), index=data.index)
df_zscore.columns = vars
df_zscore[target] = target_data
df_zscore.to_csv(f"../data/CovidPos_scaled_zscore.csv", index=False)
print(f"Data after standard scaling: {df_zscore.head(20)}")
#print(get_variable_types(df_zscore))
scaling_evaluation("../data/CovidPos_scaled_zscore.csv", "Z-Score", target)

# Approach 2 - Minmax Scaler [0,1]
transf: MinMaxScaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(data)
df_minmax = DataFrame(transf.transform(data), index=data.index)
df_minmax[target] = target_data
df_minmax.columns = vars
df_minmax.to_csv(f"../data/CovidPos_scaled_minmax.csv", index=False)
print(f"Data after minmax scaling: {df_minmax.head(20)}")
#print(get_variable_types(df_minmax))

#Approach 2 - Minmax Scaler2 [0,10]
transf: MinMaxScaler = MinMaxScaler(feature_range=(-1, 1), copy=True).fit(data)
df_minmaxRange = DataFrame(transf.transform(data), index=data.index)
df_minmaxRange[target] = target_data
df_minmaxRange.columns = vars
df_minmaxRange.to_csv(f"../data/CovidPos_scaled_minmaxRange.csv", index="id")
print(f"Data after minmax scaling: {df_minmaxRange.head(20)}")

scaling_evaluation("../data/CovidPos_scaled_minmaxRange.csv", "MinMax", target)

