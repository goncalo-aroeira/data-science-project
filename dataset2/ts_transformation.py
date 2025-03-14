from matplotlib.axes import Axes
from pandas import read_csv, DataFrame, Series, Index, Period
from matplotlib.pyplot import figure, show, savefig, tight_layout, subplots, plot, legend
from sklearn.linear_model import LinearRegression
from dslabs_functions import (plot_line_chart, HEIGHT, plot_forecasting_series, 
                              plot_forecasting_eval, series_train_test_split, ts_aggregation_by,
                              scale_all_dataframe)
from numpy import arange, array
from matplotlib.figure import Figure

#*****************************************************************************************************#
#                                           Aggregation                                               #
#*****************************************************************************************************#

def aggregation(file_tag, target, index, series, data):
    ss_mins: Series = ts_aggregation_by(series, gran_level="min", agg_func="sum")
    df_mins: DataFrame = ts_aggregation_by(data, gran_level="min", agg_func="sum")    
    df_mins.to_csv(f"data/forecast_{file_tag}_minutely.csv")

    ss_hours: Series = ts_aggregation_by(series, gran_level="H", agg_func="sum")
    df_hours: DataFrame = ts_aggregation_by(data, gran_level="H", agg_func="sum")    
    df_hours.to_csv(f"data/forecast_{file_tag}_hourly.csv")

    ss_daily: Series = ts_aggregation_by(series, gran_level="D", agg_func="sum")
    df_daily: DataFrame = ts_aggregation_by(data, gran_level="D", agg_func="sum")  
    df_daily.to_csv(f"data/forecast_{file_tag}_daily.csv")

    ss_weeks: Series = ts_aggregation_by(series, gran_level="W", agg_func="sum")
    df_weeks: DataFrame = ts_aggregation_by(data, gran_level="W", agg_func="sum")
    df_weeks.to_csv(f"data/forecast_{file_tag}_weekly.csv")


    grans: list[Series] = [ss_mins, ss_hours, ss_daily, ss_weeks]
    gran_names: list[str] = ["Minutely", "Hourly", "Daily", "Weekly"]
    filenames = ["data/forecast_fts_minutely.csv", 
                "data/forecast_fts_hourly.csv", 
                "data/forecast_fts_daily.csv", 
                "data/forecast_fts_weekly.csv"]

    print("starting loop")
    for i in range(len(grans)):
        print("starting ", filenames[i])
        aggregationStudy(filenames[i], gran_names[i], file_tag, target, index)
 
def aggregationStudy(filename: str, gran_name:str, file_tag: str, target: str, index:str):
    data: DataFrame = read_csv(filename, index_col=index, sep=",", decimal=".", parse_dates=True)
    series: Series = data[target]

    figure(figsize=(3 * HEIGHT, HEIGHT / 2))

    if gran_name == "Weekly":
        train, test = series_train_test_split(data, trn_pct=0.80)
    else: 
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
    savefig(f"images//ts_transformation/{file_tag}_linear_regression_eval_{gran_name}.png")

    print("forecast")
    plot_forecasting_series(
        train,
        test,
        prd_tst,
        title=f"{file_tag} - Linear Regression",
        xlabel=index,
        ylabel=target,
    )
    savefig(f"images//ts_transformation/{file_tag}_linear_regression_forecast_{gran_name}.png")

#*****************************************************************************************************#

#*****************************************************************************************************#
#                                                  Smoothing                                          #
#*****************************************************************************************************#
    
def smoothing(file_tag, target, index, series, data):
    window_sizes = [25, 50, 75, 100]
    filenames: list[str] = []
    for ws in window_sizes:
        df_smooth: DataFrame = data.rolling(window=ws).mean()
        df_smooth.dropna(inplace=True)
        filenames += [f"data/forecast_{file_tag}_ws_{ws}.csv"]
        df_smooth.to_csv(f"data/forecast_{file_tag}_ws_{ws}.csv")

    print(filenames)

    for i in range(len(window_sizes)):
        forecasting_after_smoothing(file_tag, target, index, filenames[i], window_sizes[i])

    
    
def forecasting_after_smoothing(file_tag, target, index, filename, window_size: int):
    data: DataFrame = read_csv(filename, index_col=index, sep=",", decimal=".", parse_dates=True)
    print(data)
    series: Series = data[target]
    train, test = series_train_test_split(data, trn_pct=0.90)
    
    trnX = arange(len(train)).reshape(-1, 1)
    trnY = train.to_numpy()
    tstX = arange(len(train), len(series)).reshape(-1, 1)
    tstY = test.to_numpy()
    
    model = LinearRegression()
    model.fit(trnX, trnY)
    prd_trn: Series = Series(model.predict(trnX), index=train.index)
    prd_tst: Series = Series(model.predict(tstX), index=test.index)
    
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
    
def differentiation(file_tag, target, index, series, data):
    ss_diff: Series = series.diff()
    ss_diff.dropna(inplace=True)
    ss_diff.to_csv(f"data/forecast_{file_tag}_first_derivative.csv")

    ss_diff2: Series = ss_diff.diff()
    ss_diff2.dropna(inplace=True)
    ss_diff2.to_csv(f"data/forecast_{file_tag}_second_derivative.csv")

    ss_list: list[Series] = [ss_diff, ss_diff2]
    names = ["FirstDerivative", "SecondDerivative"]
    filenames = [f"data/forecast_{file_tag}_first_derivative.csv", 
                 f"data/forecast_{file_tag}_second_derivative.csv"]

    for i in range(0, 2):
        forecasting_after_differentation(file_tag, target, index, filenames[i], names[i])

def forecasting_after_differentation(file_tag, target, index, filename, diff):
    data: DataFrame = read_csv(filename, index_col=index, sep=",", decimal=".", parse_dates=True)
    series: Series = data[target]
    train, test = series_train_test_split(data)
    
    trnX = arange(len(train)).reshape(-1, 1)
    trnY = train.to_numpy()
    tstX = arange(len(train), len(series)).reshape(-1, 1)
    tstY = test.to_numpy()
    
    model = LinearRegression()
    model.fit(trnX, trnY)
    prd_trn: Series = Series(model.predict(trnX), index=train.index)
    prd_tst: Series = Series(model.predict(tstX), index=test.index)
    
    plot_forecasting_eval(
        trn=train,
        tst=test,
        prd_trn=prd_trn,
        prd_tst=prd_tst,
        title=f"Forecasting eval after {diff} differentiation"
    )
    savefig(f"images/ts_transformation/{file_tag}_forecast_eval_{diff}", bbox_inches="tight")

    plot_forecasting_series(
        train,
        test,
        prd_tst,
        title=f"{file_tag} - {diff} differentiation",
        xlabel=index,
        ylabel=target,
    )
    savefig(f"images/ts_transformation/{file_tag}_forecast_ts_{diff}", bbox_inches="tight")

def scaling(file_tag, target, data):
    df: DataFrame = scale_all_dataframe(data)
    df.to_csv(f"data/forecast_{file_tag}_scaled.csv")
    ss: Series = df[target]


    df: DataFrame = scale_all_dataframe(data)

    ss: Series = df[target]
    figure(figsize=(3 * HEIGHT, HEIGHT / 2))
    plot_line_chart(
        ss.index.to_list(),
        ss.to_list(),
        xlabel=ss.index.name,
        ylabel=target,
        title=f"{file_tag} {target} after scaling",
    )
    savefig(f"images/ts_transformation/{file_tag}_{target}_scaling.png")  
    show()

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
    series: Series = data[target]
    #aggregation(file_tag, target, index, series, data)
    
    filename = "data/forecast_fts_hourly.csv"
    data: DataFrame = read_csv(
        filename, na_values="", 
        index_col=index,
        sep=",", decimal=".", 
        parse_dates=True, 
        infer_datetime_format=True
        )    
    series: Series = data[target]
    #smoothing(file_tag, target, index, series, data)

    filename = "data/forecast_fts_ws_100.csv"
    data: DataFrame = read_csv(
        filename, na_values="", 
        index_col=index,
        sep=",", decimal=".", 
        parse_dates=True, 
        infer_datetime_format=True
        )    
    series: Series = data[target]
    # differentiation(file_tag, target, index, series, data)

    filename = "data/forecast_fts_first_derivative.csv"
    data: DataFrame = read_csv(
        filename, na_values="", 
        index_col=index,
        sep=",", decimal=".", 
        parse_dates=True, 
        infer_datetime_format=True
        )    
    scaling(file_tag, target, data)


if __name__ == "__main__":
    main()