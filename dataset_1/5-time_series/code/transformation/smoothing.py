from matplotlib.pyplot import subplots, show, savefig, close
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas import read_csv, DataFrame, Series
from dslabs_functions import plot_line_chart, HEIGHT, plot_forecasting_eval, series_train_test_split, plot_forecasting_series
from sklearn.linear_model import LinearRegression
from numpy import arange


file_tag = "Covid"
target = "deaths"
data: DataFrame = read_csv(
    "../../data/forecast_Covid_weekly.csv",
    index_col="date",
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)

series: Series = data[target]

sizes: list[int] = [25, 50, 75, 100]
fig: Figure
axs: list[Axes]
fig, axs = subplots(len(sizes), 1, figsize=(3 * HEIGHT, 2*HEIGHT / 2 * len(sizes)))
fig.suptitle(f"{file_tag} {target} after smoothing")

for i in range(len(sizes)):
    ss_smooth: Series = series.rolling(window=sizes[i]).mean()
    ss_smooth.dropna(inplace=True)
    ss_smooth.to_csv(f"../../data/forecast_{file_tag}_smooth_{sizes[i]}.csv")
    plot_line_chart(
        ss_smooth.index.to_list(),
        ss_smooth.to_list(),
        ax=axs[i],
        xlabel=ss_smooth.index.name,
        ylabel=target,
        title=f"size={sizes[i]}",
    )
show()
savefig(f"images/{file_tag}_{target}_smoothing.png")


filenames: list[str] = [ "../../data/forecast_Covid_smooth_25.csv", "../../data/forecast_Covid_smooth_50.csv", "../../data/forecast_Covid_smooth_75.csv", "../../data/forecast_Covid_smooth_100.csv"]
timecol: str = "date"

for filename, size in zip(filenames, sizes):
    data: DataFrame = read_csv(filename, index_col=timecol, sep=",", decimal=".", parse_dates=True)
    series: Series = data[target]
    train, test = series_train_test_split(data, trn_pct=0.90)

    trnX = arange(len(train)).reshape(-1, 1)
    trnY = train.to_numpy()
    tstX = arange(len(train), len(data)).reshape(-1, 1)
    tstY = test.to_numpy()

    model = LinearRegression()
    model.fit(trnX, trnY)

    prd_trn: Series = Series(model.predict(trnX), index=train.index)
    prd_tst: Series = Series(model.predict(tstX), index=test.index)

    plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag}_Window_Size_{size} - Linear Regression")
    
    plot_forecasting_series(
    train,
    test,
    prd_tst,
    xlabel=timecol,
    ylabel=target,
    title = f"{file_tag} Window Size {size} - Linear Regression"
    )
