from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import figure, show, savefig, subplots, tight_layout
from dslabs_functions import plot_line_chart, ts_aggregation_by, plot_forecasting_eval, series_train_test_split, plot_forecasting_series
from sklearn.linear_model import LinearRegression
from numpy import arange

file_tag = "Covid"
target = "deaths"
data: DataFrame = read_csv(
    "../../data/forecast_Covid_scaled.csv",
    index_col="date",
    sep=",",
    decimal=".",
    parse_dates=True,
    infer_datetime_format=True,
)

series: Series = data[target]

ss_weeks: Series = ts_aggregation_by(series, gran_level="W", agg_func=sum)
df_weeks: DataFrame = ts_aggregation_by(data, gran_level="W", agg_func=sum)
df_weeks.to_csv(f"../../data/forecast_{file_tag}_weekly.csv")

ss_months: Series = ts_aggregation_by(series, gran_level="M", agg_func=sum)
df_months: DataFrame = ts_aggregation_by(data, gran_level="M", agg_func=sum)    
df_months.to_csv(f"../../data/forecast_{file_tag}_monthly.csv")

ss_quarters: Series = ts_aggregation_by(series, gran_level="Q", agg_func=sum)
df_quarters: DataFrame = ts_aggregation_by(data, gran_level="Q", agg_func=sum)  
df_quarters.to_csv(f"../../data/forecast_{file_tag}_quarterly.csv")

grans: list[Series] = [ss_weeks, ss_months, ss_quarters]
gran_names: list[str] = ["Weekly", "Monthly", "Quarterly"]

fig, axs = subplots(3, 1, figsize=(10, 15))  

for i in range(len(grans)):
    plot_line_chart(
        grans[i].index.to_list(),
        grans[i].to_list(),
        xlabel=grans[i].index.name,
        ylabel=target,
        title=f"{file_tag} {target} {gran_names[i]}",
        ax=axs[i]  
    )

tight_layout()  
savefig(f"images/{file_tag}_{target}_aggregation.png")  
show() 



filename: str = "data/time_series/ashrae.csv"
timecol: str = "date"

filenames: list[str] = [ "../../data/forecast_Covid_weekly.csv", "../../data/forecast_Covid_monthly.csv", "../../data/forecast_Covid_quarterly.csv"]


for filename, gran_name in zip(filenames, gran_names):
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

    plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag}_{gran_name} - Linear Regression")
    
    plot_forecasting_series(
    train,
    test,
    prd_tst,
    xlabel=timecol,
    ylabel=target,
    title = f"{file_tag} {gran_name} - Linear Regression"
    )
