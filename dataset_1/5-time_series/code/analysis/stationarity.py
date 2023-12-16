from pandas import Series
from matplotlib.pyplot import show, savefig, figure, plot, legend
from statsmodels.tsa.seasonal import DecomposeResult, seasonal_decompose
from dslabs_functions import plot_components, plot_line_chart, HEIGHT
from pandas import DataFrame, Series, read_csv


file_tag = "Covid"
filename = "../../data/forecast_covid_single.csv"
index = "date"
target = "deaths"

data: DataFrame = read_csv(
    filename,
    index_col=index,
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)
series: Series = data[target]

plot_components(
    series,
    title=f"{file_tag} weekly {target}",
    x_label=series.index.name,
    y_label=target,
)
show()
savefig(f"../images/{file_tag}_components_study.png")



figure(figsize=(3 * HEIGHT, HEIGHT))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{file_tag} stationary study",
    name="original",
)
n: int = len(series)
plot(series.index, [series.mean()] * n, "r-", label="mean")
legend()
show()
savefig(f"../images/{file_tag}_stationarity_study_1.png")


BINS = 10
mean_line: list[float] = []

for i in range(BINS):
    segment: Series = series[i * n // BINS : (i + 1) * n // BINS]
    mean_value: list[float] = [segment.mean()] * (n // BINS)
    mean_line += mean_value
mean_line += [mean_line[-1]] * (n - len(mean_line))

figure(figsize=(3 * HEIGHT, HEIGHT))
plot_line_chart(
    series.index.to_list(),
    series.to_list(),
    xlabel=series.index.name,
    ylabel=target,
    title=f"{file_tag} stationary study",
    name="original",
    show_stdev=True,
)
n: int = len(series)
plot(series.index, mean_line, "r-", label="mean")
legend()
show()
savefig(f"../images/{file_tag}_stationarity_study_2.png")


from statsmodels.tsa.stattools import adfuller


def eval_stationarity(series: Series) -> bool:
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]:.3f}")
    print(f"p-value: {result[1]:.3f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"\t{key}: {value:.3f}")
    return result[1] <= 0.05


print(f"The series {('is' if eval_stationarity(series) else 'is not')} stationary")