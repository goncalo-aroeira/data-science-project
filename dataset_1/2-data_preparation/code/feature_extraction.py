from sklearn.decomposition import PCA
from pandas import Series, Index, DataFrame, read_csv
from matplotlib.axes import Axes
from dslabs_functions import plot_bar_chart
from matplotlib.pyplot import figure, show, savefig, gca


data_filename: str = "../data/CovidPos_scaled_minmax.csv"
data: DataFrame = read_csv(data_filename)
target = "CovidPos"

target_data: Series = data.pop(target)
index: Index = data.index
pca = PCA()
pca.fit(data)

xvalues: list[str] = [f"PC{i+1}" for i in range(len(pca.components_))]
print(pca.feature_names_in_)
figure(figsize=(15, 5))
ax: Axes = gca()
plot_bar_chart(
    xvalues,
    pca.explained_variance_ratio_,
    ax=ax,
    title="Explained variance ratio",
    xlabel="PC",
    ylabel="ratio",
    percentage=True,
)
ax.plot(pca.explained_variance_ratio_)
savefig("../images/CovidPos_pca_explained_variance_ratio.png")
show()