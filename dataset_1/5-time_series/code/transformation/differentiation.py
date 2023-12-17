from dslabs_functions import plot_line_chart
from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show, savefig, subplots, tight_layout
from dslabs_functions import plot_line_chart, ts_aggregation_by, HEIGHT, plot_forecasting_eval, series_train_test_split, plot_forecasting_series
from sklearn.linear_model import LinearRegression
from numpy import arange

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
ss_diff.to_csv(f"../../data/forecast_{file_tag}_diff.csv")

figure(figsize=(3 * HEIGHT, HEIGHT))
plot_line_chart(
    ss_diff.index.to_list(),
    ss_diff.to_list(),
    title="Differentiation",
    xlabel=series.index.name,
    ylabel=target,
)
show()

filename = "../../data/forecast_Covid_diff.csv"
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

plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag}_Diff - Linear Regression")

plot_forecasting_series(
train,
test,
prd_tst,
xlabel=timecol,
ylabel=target,
title = f"{file_tag} Diff - Linear Regression"
)
