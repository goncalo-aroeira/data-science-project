from matplotlib.axes import Axes
from pandas import read_csv, DataFrame, Series, Index, Period
from matplotlib.pyplot import figure, show, savefig, tight_layout, subplots, plot, legend
from sklearn.linear_model import LinearRegression
from sklearn.base import RegressorMixin
from dslabs_functions import (plot_line_chart, HEIGHT, plot_forecasting_series, 
                              plot_forecasting_eval, series_train_test_split, 
                              FORECAST_MEASURES, DELTA_IMPROVE)
from numpy import mean
from matplotlib.figure import Figure

#*****************************************************************************************************#
#                                        Average Regressor                                            #
#*****************************************************************************************************#
class SimpleAvgRegressor(RegressorMixin):
    def __init__(self):
        super().__init__()
        self.mean: float = 0.0
        return

    def fit(self, X: Series):
        self.mean = X.mean()
        return

    def predict(self, X: Series) -> Series:
        prd: list = len(X) * [self.mean]
        prd_series: Series = Series(prd)
        prd_series.index = X.index
        return prd_series
    
def simpleAverage(file_tag, target, timecol, train, test):
    fr_mod = SimpleAvgRegressor()
    fr_mod.fit(train)
    prd_trn: Series = fr_mod.predict(train)
    prd_tst: Series = fr_mod.predict(test)

    plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag} - Simple Average")
    savefig(f"images/ts_forecasting/{file_tag}_simpleAvg_eval.png")

    plot_forecasting_series(
        train,
        test,
        prd_tst,
        title=f"{file_tag} - Simple Average",
        xlabel=timecol,
        ylabel=target,
    )
    savefig(f"images/ts_forecasting/{file_tag}_simpleAvg_forecast.png")

#*****************************************************************************************************#
#                                     Peresistence Regressor                                          #
#*****************************************************************************************************#
class PersistenceOptimistRegressor(RegressorMixin):
    def __init__(self):
        super().__init__()
        self.last: float = 0.0
        return

    def fit(self, X: Series):
        self.last = X.iloc[-1]
        # print(self.last)
        return

    def predict(self, X: Series):
        prd: list = X.shift().values.ravel()
        prd[0] = self.last
        prd_series: Series = Series(prd)
        prd_series.index = X.index
        return prd_series
    
class PersistenceRealistRegressor(RegressorMixin):
    def __init__(self):
        super().__init__()
        self.last = 0
        self.estimations = [0]
        self.obs_len = 0

    def fit(self, X: Series):
        for i in range(1, len(X)):
            self.estimations.append(X.iloc[i - 1])
        self.obs_len = len(self.estimations)
        self.last = X.iloc[len(X) - 1]
        prd_series: Series = Series(self.estimations)
        prd_series.index = X.index
        return prd_series

    def predict(self, X: Series):
        prd: list = len(X) * [self.last]
        prd_series: Series = Series(prd)
        prd_series.index = X.index
        return prd_series
    
def persistenceRegressor(file_tag, target, timecol, train, test):
    #Optimist
    fr_mod = PersistenceOptimistRegressor()
    fr_mod.fit(train)
    prd_trn: Series = fr_mod.predict(train)
    prd_tst: Series = fr_mod.predict(test)

    plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag} - Persistence Optimist")
    savefig(f"images/ts_forecasting/{file_tag}_persistence_optim_eval.png")

    plot_forecasting_series(
        train,
        test,
        prd_tst,
        title=f"{file_tag} - Persistence Optimist",
        xlabel=timecol,
        ylabel=target,
    )
    savefig(f"images/ts_forecasting/{file_tag}_persistence_optim_forecast.png")

    #Realist
    fr_mod = PersistenceRealistRegressor()
    fr_mod.fit(train)
    prd_trn: Series = fr_mod.predict(train)
    prd_tst: Series = fr_mod.predict(test)

    plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag} - Persistence Realist")
    savefig(f"images/ts_forecasting/{file_tag}_persistence_real_eval.png")

    plot_forecasting_series(
        train,
        test,
        prd_tst,
        title=f"{file_tag} - Persistence Realist",
        xlabel=timecol,
        ylabel=target,
    )
    savefig(f"images/ts_forecasting/{file_tag}_persistence_real_forecast.png")

#*****************************************************************************************************#
#                                           Rolling Mean                                               #
#*****************************************************************************************************#

class RollingMeanRegressor(RegressorMixin):
    def __init__(self, win: int = 3):
        super().__init__()
        self.win_size = win
        self.memory: list = []

    def fit(self, X: Series):
        self.memory = X.iloc[-self.win_size :]
        # print(self.memory)
        return

    def predict(self, X: Series):
        estimations = self.memory.tolist()
        for i in range(len(X)):
            new_value = mean(estimations[len(estimations) - self.win_size - i :])
            estimations.append(new_value)
        prd_series: Series = Series(estimations[self.win_size :])
        prd_series.index = X.index
        return prd_series
    
def rolling_mean_study(train: Series, test: Series, measure: str = "R2"):
    # win_size = (3, 5, 10, 15, 20, 25, 30, 40, 50)
    win_size = (12, 24, 48, 96, 192, 384, 768)
    flag = measure == "R2" or measure == "MAPE"
    best_model = None
    best_params: dict = {"name": "Rolling Mean", "metric": measure, "params": ()}
    best_performance: float = -100000

    print("train",train)
    print("test",test)
    print("flag",flag)

    yvalues = []
    for w in win_size:
        pred = RollingMeanRegressor(win=w)
        pred.fit(train)
        prd_tst = pred.predict(test)

        eval: float = FORECAST_MEASURES[measure](test, prd_tst)
        # print(w, eval)
        if eval > best_performance and abs(eval - best_performance) > DELTA_IMPROVE:
            best_performance: float = eval
            best_params["params"] = (w,)
            best_model = pred
        yvalues.append(eval)

    print(f"Rolling Mean best with win={best_params['params'][0]:.0f} -> {measure}={best_performance}")
    plot_line_chart(
        win_size, yvalues, title=f"Rolling Mean ({measure})", xlabel="window size", ylabel=measure, percentage=flag
    )

    return best_model, best_params

def rollingMean(file_tag, target, timecol, measure, train, test):
    fig = figure(figsize=(HEIGHT, HEIGHT))
    best_model, best_params = rolling_mean_study(train, test)
    savefig(f"images/ts_forecasting/{file_tag}_rollingmean_{measure}_study.png")

    params = best_params["params"]
    prd_trn: Series = best_model.predict(train)
    prd_tst: Series = best_model.predict(test)

    plot_forecasting_eval(train, test, prd_trn, prd_tst, title=f"{file_tag} - Rolling Mean (win={params[0]})")
    savefig(f"images/ts_forecasting/{file_tag}_rollingmean_{measure}_win{params[0]}_eval.png")
#*****************************************************************************************************#


def main():
    filename = "data/forecast_fts_scaled.csv"
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

    train, test = series_train_test_split(data, trn_pct=0.90)
    # simpleAverage(file_tag, target, index, train, test)
    # persistenceRegressor(file_tag, target, index, train, test)
    measure: str = "R2"
    rollingMean(file_tag, target, index, measure, train, test)

if __name__ == "__main__":
    main()