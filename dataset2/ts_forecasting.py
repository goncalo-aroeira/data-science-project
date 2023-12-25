from matplotlib.axes import Axes
from pandas import read_csv, DataFrame, Series, Index, Period
from matplotlib.pyplot import figure, show, savefig, tight_layout, subplots, plot, legend
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.base import RegressorMixin
from dslabs_functions import (plot_line_chart, HEIGHT, plot_forecasting_series, 
                              plot_forecasting_eval, series_train_test_split, 
                              FORECAST_MEASURES, DELTA_IMPROVE, plot_multiline_chart)
from numpy import mean
from matplotlib.figure import Figure
from torch import no_grad, tensor
from torch.nn import LSTM, Linear, Module, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy

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
    win_size = (3, 5, 10, 15, 20, 25, 30, 40, 50)
    #win_size = (12, 24, 48, 96, 192, 384, 768)
    flag = measure == "R2" or measure == "MAPE"
    best_model = None
    best_params: dict = {"name": "Rolling Mean", "metric": measure, "params": ()}
    best_performance: float = -100000

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

    plot_forecasting_series(
        train,
        test,
        prd_tst,
        title=f"{file_tag} - Rolling Mean (win={params[0]})",
        xlabel=timecol,
        ylabel=target,
    )
    savefig(f"images/ts_forecasting/{file_tag}_rollingmean_{measure}_forecast.png")

#*****************************************************************************************************#
#                                              ARIMA                                                  #
#*****************************************************************************************************#
def arima_study(train: Series, test: Series, measure: str = "R2"):
    d_values = (0, 1, 2)
    p_params = (1, 2, 3, 5, 7, 10)
    q_params = (1, 3, 5, 7)

    flag = measure == "R2" or measure == "MAPE"
    best_model = None
    best_params: dict = {"name": "ARIMA", "metric": measure, "params": ()}
    best_performance: float = -100000

    fig, axs = subplots(1, len(d_values), figsize=(len(d_values) * HEIGHT, HEIGHT))
    for i in range(len(d_values)):
        d: int = d_values[i]
        values = {}
        for q in q_params:
            yvalues = []
            for p in p_params:
                arima = ARIMA(train, order=(p, d, q))
                model = arima.fit()
                prd_tst = model.forecast(steps=len(test), signal_only=False)
                eval: float = FORECAST_MEASURES[measure](test, prd_tst)
                # print(f"ARIMA ({p}, {d}, {q})", eval)
                if eval > best_performance and abs(eval - best_performance) > DELTA_IMPROVE:
                    best_performance: float = eval
                    best_params["params"] = (p, d, q)
                    best_model = model
                yvalues.append(eval)
            values[q] = yvalues
        plot_multiline_chart(
            p_params, values, ax=axs[i], title=f"ARIMA d={d} ({measure})", xlabel="p", ylabel=measure, percentage=flag
        )
    print(
        f"ARIMA best results achieved with (p,d,q)=({best_params['params'][0]:.0f}, {best_params['params'][1]:.0f}, {best_params['params'][2]:.0f}) ==> measure={best_performance:.2f}"
    )

    return best_model, best_params


def arima(file_tag, target, timecol, measure, train, test):

    predictor = ARIMA(train, order=(3, 1, 2))
    model = predictor.fit()
    print(model.summary())
    model.plot_diagnostics(figsize=(2 * HEIGHT, 1.5 * HEIGHT))


    best_model, best_params = arima_study(train, test, measure=measure)
    savefig(f"images/ts_forecasting/{file_tag}_arima_{measure}_study.png")

    params = best_params["params"]
    prd_trn = best_model.predict(start=0, end=len(train) - 1)
    prd_tst = best_model.forecast(steps=len(test))

    plot_forecasting_eval(
        train, test, prd_trn, prd_tst, title=f"{file_tag} - ARIMA (p={params[0]}, d={params[1]}, q={params[2]})"
    )
    savefig(f"images/ts_forecasting/{file_tag}_arima_{measure}_eval.png")

    plot_forecasting_series(
        train,
        test,
        prd_tst,
        title=f"{file_tag} - ARIMA ",
        xlabel=timecol,
        ylabel=target,
    )
    savefig(f"images/ts_forecasting/{file_tag}_arima_{measure}_forecast.png")

#*****************************************************************************************************#
#                                              LSTMs                                                  #
#*****************************************************************************************************#
def prepare_dataset_for_lstm(series, seq_length: int = 4):
    setX: list = []
    setY: list = []
    for i in range(len(series) - seq_length):
        past = series[i : i + seq_length]
        future = series[i + 1 : i + seq_length + 1]
        setX.append(past)
        setY.append(future)
    return tensor(setX), tensor(setY)


class DS_LSTM(Module):
    def __init__(self, train, input_size: int = 1, hidden_size: int = 50, num_layers: int = 1, length: int = 4):
        super().__init__()
        self.lstm = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = Linear(hidden_size, 1)
        self.optimizer = Adam(self.parameters())
        self.loss_fn = MSELoss()

        trnX, trnY = prepare_dataset_for_lstm(train, seq_length=length)
        self.loader = DataLoader(TensorDataset(trnX, trnY), shuffle=True, batch_size=len(train) // 10)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

    def fit(self):
        self.train()
        for batchX, batchY in self.loader:
            y_pred = self(batchX)
            loss = self.loss_fn(y_pred, batchY)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss

    def predict(self, X):
        with no_grad():
            y_pred = self(X)
        return y_pred[:, -1, :]
    
def lstm_study(train, test, file_tag:str,  nr_episodes: int = 1000, measure: str = "R2"):
    sequence_size = [2, 4, 8]
    nr_hidden_units = [25, 50, 100]

    step: int = nr_episodes // 10
    episodes = [1] + list(range(0, nr_episodes + 1, step))[1:]
    flag = measure == "R2" or measure == "MAPE"
    best_model = None
    best_params: dict = {"name": "LSTM", "metric": measure, "params": ()}
    best_performance: float = -100000

    _, axs = subplots(1, len(sequence_size), figsize=(len(sequence_size) * HEIGHT, HEIGHT))

    for i in range(len(sequence_size)):
        length = sequence_size[i]
        tstX, tstY = prepare_dataset_for_lstm(test, seq_length=length)

        values = {}
        for hidden in nr_hidden_units:
            yvalues = []
            model = DS_LSTM(train, hidden_size=hidden)
            for n in range(0, nr_episodes + 1):
                model.fit()
                if n % step == 0:
                    prd_tst = model.predict(tstX)
                    eval: float = FORECAST_MEASURES[measure](test[length:], prd_tst)
                    print(f"seq length={length} hidden_units={hidden} nr_episodes={n}", eval)
                    if eval > best_performance and abs(eval - best_performance) > DELTA_IMPROVE:
                        best_performance: float = eval
                        best_params["params"] = (length, hidden, n)
                        best_model = deepcopy(model)
                    yvalues.append(eval)
            values[hidden] = yvalues
        plot_multiline_chart(
            episodes,
            values,
            ax=axs[i],
            title=f"LSTM seq length={length} ({measure})",
            xlabel="nr episodes",
            ylabel=measure,
            percentage=flag,
        )
    savefig(f"images/ts_forecasting/{file_tag}_lstm_study_{measure}_{length}.png")
    print(
        f"LSTM best results achieved with length={best_params['params'][0]} hidden_units={best_params['params'][1]} and nr_episodes={best_params['params'][2]}) ==> measure={best_performance:.2f}"
    )
    return best_model, best_params


    
def lstm(file_tag, target, timecol, measure, train, test, data):
    measure: str = "R2"

    series = data[[target]].values.astype("float32")

    train_size = int(len(series) * 0.90)
    train, test = series[:train_size], series[train_size:]

    model = DS_LSTM(train, input_size=1, hidden_size=50, num_layers=1)
    loss = model.fit()
    print(loss)
    
    best_model, best_params = lstm_study(train, test, file_tag, nr_episodes=3000, measure=measure)

    params = best_params["params"]
    best_length = params[0]
    trnX, trnY = prepare_dataset_for_lstm(train, seq_length=best_length)
    tstX, tstY = prepare_dataset_for_lstm(test, seq_length=best_length)

    prd_trn = best_model.predict(trnX)
    prd_tst = best_model.predict(tstX)

    plot_forecasting_eval(
        train[best_length:],
        test[best_length:],
        prd_trn,
        prd_tst,
        title=f"{file_tag} - LSTM (length={best_length}, hidden={params[1]}, epochs={params[2]})",
    )
    savefig(f"images/ts_forecasting/{file_tag}_lstms_{measure}_eval.png")


    series = data[[target]]
    train, test = series[:train_size], series[train_size:]
    pred_series: Series = Series(prd_tst.numpy().ravel(), index=test.index[best_length:])

    plot_forecasting_series(
        train[best_length:],
        test[best_length:],
        pred_series,
        title=f"{file_tag} - LSTMs ",
        xlabel=timecol,
        ylabel=target,
    )
    savefig(f"images/ts_forecasting/{file_tag}_lstms_{measure}_forecast.png")

#*****************************************************************************************************#


def main():
    filename = "data/forecast_fts_scaled.csv"
    file_tag = "fts"
    target = "Total"
    index = timecol = "Timestamp"
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
    # measures  =["R2", "MAPE"]
    # rollingMean(file_tag, target, index, measure, train, test)
    # arima(file_tag, target, timecol, measure, train, test)
    lstm(file_tag, target, timecol, measure, train, test, data)

if __name__ == "__main__":
    main()