from matplotlib.axes import Axes
from pandas import read_csv, DataFrame, Series, Index, Period
from matplotlib.pyplot import figure, show, savefig, tight_layout, subplots, plot, legend
from sklearn.linear_model import LinearRegression
from dslabs_functions import (plot_line_chart, HEIGHT, plot_forecasting_series, 
                              plot_forecasting_eval, series_train_test_split, ts_aggregation_by)
from numpy import arange, array
from matplotlib.figure import Figure

#*****************************************************************************************************#
#                                           Aggregation                                               #
#*****************************************************************************************************#
 
def aggregationStudy(filename: str, gran_name:str, file_tag: str, target: str, index:str):
    data: DataFrame = read_csv(filename, index_col=index, sep=",", decimal=".", parse_dates=True)
    series: Series = data[target]

    figure(figsize=(3 * HEIGHT, HEIGHT / 2))

    if gran_name == "Weekly":
        train, test = series_train_test_split(data, trn_pct=0.80)
    train, test = series_train_test_split(data, trn_pct=0.90)
    print("\ntrain\n",train, "\ntest\n",test)

    trnX = arange(len(train)).reshape(-1, 1)
    trnY = train.to_numpy()
    tstX = arange(len(train), len(data)).reshape(-1, 1)
    tstY = test.to_numpy()

    model = LinearRegression()
    model.fit(trnX, trnY)

    prd_trn: Series = Series(model.predict(trnX), index=train.index)
    prd_tst: Series = Series(model.predict(tstX), index=test.index)

    print("evaluating")
    plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag} - Linear Regression")
    savefig(f"images/{file_tag}_linear_regression_eval_{gran_name}.png")

    print("forecast")
    plot_forecasting_series(
        train,
        test,
        prd_tst,
        title=f"{file_tag} - Linear Regression",
        xlabel=index,
        ylabel=target,
    )
    savefig(f"images/{file_tag}_linear_regression_forecast_{gran_name}.png")

#***********************************************************************************

#*****************************************************************************************************#

#*****************************************************************************************************#
#                                                  Smoothing                                          #
#*****************************************************************************************************#
    
def forecasting_after_smoothing(file_tag, target, index, window_size: int, series: Series):
    ss_smooth: Series = series.rolling(window=window_size).mean()
    train, test = series_train_test_split(ss_smooth)
    
    trnX = arange(len(train)).reshape(-1, 1)
    trnY = train.to_numpy()
    tstX = arange(len(train), len(series)).reshape(-1, 1)
    tstY = test.to_numpy()
    
    model = LinearRegression()
    model.fit(trnX, trnY)
    prd_trn: Series = Series(model.predict(trnX))
    prd_tst: Series = Series(model.predict(tstX))
    
    plot_forecasting_eval(
        trn=train,
        tst=test,
        prd_trn=prd_trn,
        prd_tst=prd_tst,
        title=f"Forecasting eval after {window_size} smoothing"
    )
    savefig(f"images/ts_transformation/{file_tag}_forecast_eval_after_smooth_ws_{window_size}", bbox_inches="tight")

    plot_forecasting_series(
        train,
        test,
        prd_tst,
        title=f"{file_tag} - {window_size} smoothing",
        xlabel=index,
        ylabel=target,
    )
    savefig(f"images/ts_transformation/{file_tag}_forecast_ts_after_smooth_ws_{window_size}", bbox_inches="tight")

#*****************************************************************************************************#
    
#*****************************************************************************************************#
#                                            Differentiation                                          #
#*****************************************************************************************************#

def forecasting_after_differentation(file_tag, target, index, diff, series: Series):
    ss_diff: Series = series.diff()
    train, test = series_train_test_split(ss_diff)
    
    trnX = arange(len(train)).reshape(-1, 1)
    trnY = train.to_numpy()
    tstX = arange(len(train), len(series)).reshape(-1, 1)
    tstY = test.to_numpy()
    
    model = LinearRegression()
    model.fit(trnX, trnY)
    prd_trn: Series = Series(model.predict(trnX))
    prd_tst: Series = Series(model.predict(tstX))
    
    plot_forecasting_eval(
        trn=train,
        tst=test,
        prd_trn=prd_trn,
        prd_tst=prd_tst,
        title=f"Forecasting eval after {diff} differentiation"
    )
    savefig(f"images/ts_transformation/{file_tag}_forecast_eval_after_diff_{diff}", bbox_inches="tight")

    plot_forecasting_series(
        train,
        test,
        prd_tst,
        title=f"{file_tag} - {diff} differentiation",
        xlabel=index,
        ylabel=target,
    )
    savefig(f"images/ts_transformation/{file_tag}_forecast_ts_after_diff_{diff}", bbox_inches="tight")

#*****************************************************************************************************#


def main():
    filename = "data/forecast_traffic_single.csv"
    file_tag = "fts"
    target = "Total"
    index = "Timestamp"
    data: DataFrame = read_csv(
        filename, na_values="", 
        index_col=index,
        sep=",", decimal=".", 
        parse_dates=True, 
        infer_datetime_format=True
        )
    
    stroke: DataFrame = read_csv(filename, na_values="")
    series: Series = data[target]

    ss_mins: Series = ts_aggregation_by(series, gran_level="min", agg_func="mean")
    df_mins: DataFrame = ts_aggregation_by(data, gran_level="min", agg_func="mean")    
    df_mins.to_csv(f"data/forecast_{file_tag}_minutely.csv")

    ss_hours: Series = ts_aggregation_by(series, gran_level="H", agg_func="mean")
    df_hours: DataFrame = ts_aggregation_by(data, gran_level="H", agg_func="mean")    
    df_hours.to_csv(f"data/forecast_{file_tag}_hourly.csv")

    ss_daily: Series = ts_aggregation_by(series, gran_level="D", agg_func="mean")
    df_daily: DataFrame = ts_aggregation_by(data, gran_level="D", agg_func="mean")  
    df_daily.to_csv(f"data/forecast_{file_tag}_daily.csv")

    ss_weeks: Series = ts_aggregation_by(series, gran_level="W", agg_func="mean")
    df_weeks: DataFrame = ts_aggregation_by(data, gran_level="W", agg_func="mean")
    df_weeks.to_csv(f"data/forecast_{file_tag}_weekly.csv")


    grans: list[Series] = [ss_mins, ss_hours, ss_daily, ss_weeks]
    gran_names: list[str] = ["Minutely", "Hourly", "Daily", "Weekly"]
    filenames = ["data/forecast_fts_minutely.csv", 
                "data/forecast_fts_hourly.csv", 
                "data/forecast_fts_daily.csv", 
                "data/forecast_fts_weekly.csv"]

    # grans: list[str] = ["min", "H","D", "W"]

    print("starting loop")
    for i in [3]:
    # for i in range(len(grans)):
        print("starting ", filenames[i])
        aggregationStudy
        


if __name__ == "__main__":
    main()