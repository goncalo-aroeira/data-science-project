from dslabs_functions import plot_line_chart
from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show, savefig, subplots, tight_layout
from dslabs_functions import plot_line_chart, ts_aggregation_by, HEIGHT, plot_forecasting_eval, series_train_test_split, plot_forecasting_series
from sklearn.linear_model import LinearRegression
from numpy import arange
from matplotlib.axes import Axes
from matplotlib.figure import Figure

filename = "../../data/forecast_Covid_smooth_100.csv"
file_tag = "Covid"
target = "deaths"
timecol = "date"

data: DataFrame = read_csv(
    filename,
    index_col="date",
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)

series: Series = data[target]

ss_diff: Series = series.diff()
ss_diff.dropna(inplace=True)
ss_diff.to_csv(f"../../data/forecast_{file_tag}_first_derivative.csv")

ss_diff2: Series = ss_diff.diff()
ss_diff2.dropna(inplace=True)
ss_diff2.to_csv(f"../../data/forecast_{file_tag}_second_derivative.csv")

ss_list: list[Series] = [ss_diff, ss_diff2]
names = ["First derivative", "Second derivative"]

fig: Figure
axs: list[Axes]
fig, axs = subplots(len(ss_list), 1, figsize=(3 * HEIGHT, 2*HEIGHT / 2 * len(names)))
fig.suptitle(f"{file_tag} {target} after derivate")

for i in range(len(ss_list)):
    plot_line_chart(
        ss_list[i].index.to_list(),
        ss_list[i].to_list(),
        ax=axs[i],
        xlabel=ss_list[i].index.name,
        ylabel=target,
        title=f"{names[i]}",
    )
show()
savefig(f"images/{file_tag}_{target}_differentiation.png")



filenames = [ "../../data/forecast_Covid_first_derivative.csv", "../../data/forecast_Covid_second_derivative.csv"]

for filename, name in zip(filenames, names):
    data: DataFrame = read_csv(filename, index_col=timecol, sep=",", decimal=".", parse_dates=True)
    series: Series = data[target]
    train, test = series_train_test_split(data, trn_pct=0.90)
    
    train.to_csv(f"../../data/forecast_{file_tag}_train.csv")
    test.to_csv(f"../../data/forecast_{file_tag}_test.csv")

    trnX = arange(len(train)).reshape(-1, 1)
    trnY = train.to_numpy()
    tstX = arange(len(train), len(data)).reshape(-1, 1)
    tstY = test.to_numpy()

    model = LinearRegression()
    model.fit(trnX, trnY)

    prd_trn: Series = Series(model.predict(trnX), index=train.index)
    prd_tst: Series = Series(model.predict(tstX), index=test.index)

    plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag}_{name} - Linear Regression")

    plot_forecasting_series(
    train,
    test,
    prd_tst,
    xlabel=timecol,
    ylabel=target,
    title = f"{file_tag} {name} - Linear Regression"
    )
    