from pandas import read_csv, DataFrame, Series
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.pyplot import figure, savefig, close, show, subplots

data_filename: str = "../data/CovidPos_outliers_rowDrop_stdBased.csv"
file_tag: str = "CovidPos"
target = "CovidPos"

data: DataFrame = read_csv(data_filename)

# Approach 1 - Standard scaling (z-score transformation)
vars: list[str] = data.columns.to_list()
target_data: Series = data.pop(file_tag)

transf: StandardScaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(
    data
)
df_zscore = DataFrame(transf.transform(data), index=data.index)
df_zscore[file_tag] = target_data
df_zscore.columns = vars
df_zscore.to_csv(f"../data/{file_tag}_scaled_zscore.csv", index="ID")
print(f"Data after standard scaling: {df_zscore.head(20)}")

# Approach 2 - Minmax Scaler [0,1]
transf: MinMaxScaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(data)
df_minmax = DataFrame(transf.transform(data), index=data.index)
df_minmax[target] = target_data
df_minmax.columns = vars
df_minmax.to_csv(f"../data/{file_tag}_scaled_minmax.csv", index="id")
print(f"Data after minmax scaling: {df_minmax.head(20)}")

# Evaluate the different approaches
fig, axs = subplots(1, 3, figsize=(25, 10), squeeze=False)
axs[0, 0].set_title("Original data")
data.boxplot(ax=axs[0, 0])
axs[0, 1].set_title("Z-score normalization")
df_zscore.boxplot(ax=axs[0, 1])
# axs[0, 2].set_title("MinMax2 normalization")
# df_minmaxRange.boxplot(ax=axs[0, 2])
axs[0, 2].set_title("MinMax normalization")
df_minmax.boxplot(ax=axs[0, 2])
savefig(f"../images/CovidPos_scaling.png")
show()