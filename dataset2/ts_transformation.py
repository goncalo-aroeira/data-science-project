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
 
def forecasting_after_aggregation(file_tag, target, index, gran_level: str, data: DataFrame):
    ss_agg: Series = ts_aggregation_by(data, gran_level=gran_level, agg_func="sum")
    train, test = series_train_test_split(ss_agg)
    
    trnX = arange(len(train)).reshape(-1, 1)
    trnY = train.to_numpy()
    tstX = arange(len(train), len(data)).reshape(-1, 1)
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
        title=f"Forecasting eval after {gran_level} aggregation"
    )
    savefig(f"images/ts_transformation/{file_tag}_forecast_eval_after_{gran_level}_aggro.png", bbox_inches="tight")

    plot_forecasting_series(
        train,
        test,
        prd_tst,
        title=f"{file_tag} - {gran_level} aggregation",
        xlabel=index,
        ylabel=target,
    )
    savefig(f"images/ts_transformation/{file_tag}_forecast_ts_after_{gran_level}_aggro.png", bbox_inches="tight")

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
    series: Series = data[target]

    # Aggregation results
    #for gran in ["min", "H", "D", "W", "M", "Q"]:
    #    forecasting_after_aggregation(file_tag, target, index, gran, data)

    # Smoothing results
    #for ws in [25, 50, 75, 100]:
    #    forecasting_after_smoothing(file_tag, target, index, ws, series)

    # Differentiation results
    for diff in ["d1", "d2"]:
        forecasting_after_differentation(file_tag, target, index, diff, series)
        


if __name__ == "__main__":
    main()
