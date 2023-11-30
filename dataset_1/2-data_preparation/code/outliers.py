from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, savefig, close, show
from dslabs_functions import determine_outlier_thresholds_for_var, plot_multibar_chart, evaluate_approach
from sklearn.model_selection import train_test_split

def outliers_evaluation(data_filename: str, strategy: str):
    data: DataFrame = read_csv(data_filename)

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    ## Evaluate Approach
    file_tag = "CovidPos"
    target = "CovidPos"

    figure()
    eval: dict[str, list] = evaluate_approach(train_data, test_data, target=target, metric="recall")
    plot_multibar_chart(
        ["NB", "KNN"], eval, title=f"{file_tag} evaluation", percentage=True)

    savefig(f"../images/CovidPos_outliers_treat_{strategy}.png")
    show()
    
    
target = "CovidPos"


data: DataFrame = read_csv("../data/ccs_mvi_fill_frequent.csv")
# Approach 1 - Dropping outliers (std_based = true)
print(f"Data before dropping outliers: {data.shape}")
data_outliers_rowDrop: DataFrame = data.copy(deep=True)
summary5: DataFrame = data.describe()
for var in data.columns:
    top_threshold, bottom_threshold = determine_outlier_thresholds_for_var(summary5[var])
    outliers: Series = data_outliers_rowDrop[(data_outliers_rowDrop[var] > top_threshold) | (data_outliers_rowDrop[var] < bottom_threshold)]
    data_outliers_rowDrop.drop(outliers.index, axis=0, inplace=True)
data_outliers_rowDrop.to_csv("../data/CovidPos_outliers_rowDrop_stdBased.csv")
print(f"Data after dropping outliers: {data_outliers_rowDrop.shape}")
outliers_evaluation("../data/CovidPos_outliers_rowDrop_stdBased.csv", "rowDrop_stdBased")


# Approach 2 - Dropping outliers (std_based = false)
print(f"Data before dropping outliers: {data.shape}")
data_outliers_rowDrop: DataFrame = data.copy(deep=True)
summary5: DataFrame = data.describe()
for var in data.columns:
    top_threshold, bottom_threshold = determine_outlier_thresholds_for_var(summary5[var], std_based=False)
    outliers: Series = data_outliers_rowDrop[(data_outliers_rowDrop[var] > top_threshold) | (data_outliers_rowDrop[var] < bottom_threshold)]
    data_outliers_rowDrop.drop(outliers.index, axis=0, inplace=True)
data_outliers_rowDrop.to_csv("../data/CovidPos_outliers_rowDrop_NotStdBased.csv")
print(f"Data after dropping outliers: {data_outliers_rowDrop.shape}")
outliers_evaluation("../data/CovidPos_outliers_rowDrop_NotStdBased.csv", "rowDrop_NotStdBased")


# Approach 3 - Replacing Outliers (if > than max or < min) with fixed values - using the median
print(f"Data before replacing outliers: {data.shape}")
data_outliers_rep_fixedMedian: DataFrame = data.copy(deep=True)
for var in data.columns:
    top, bottom = determine_outlier_thresholds_for_var(summary5[var])
    median: float = data_outliers_rep_fixedMedian[var].median()
    data_outliers_rep_fixedMedian[var] = data_outliers_rep_fixedMedian[var].apply(lambda x: median if x > top or x < bottom else x)
data_outliers_rep_fixedMedian.to_csv("../data/CovidPos_outliers_rep_fixed_median.csv")
print(f"Data after replacing outliers: {data_outliers_rep_fixedMedian.shape}")
outliers_evaluation("../data/CovidPos_outliers_rep_fixed_median.csv", "rep_fixed_median")


# Approach 4 - Truncating outliers with the max and min
print(f"Data before truncating outliers: {data.shape}")
data_outliers_trunc: DataFrame = data.copy(deep=True)
for var in data.columns:
    top, bottom = determine_outlier_thresholds_for_var(summary5[var])
    data_outliers_trunc[var] = data_outliers_trunc[var].apply(lambda x: top if x > top else bottom if x < bottom else x)
data_outliers_trunc.to_csv("../data/CovidPos_outliers_trunc_minmax.csv")
print(f"Data after truncating outliers: {data_outliers_trunc.shape}")
outliers_evaluation("../data/CovidPos_outliers_trunc_minmax.csv", "truncating_minmax")