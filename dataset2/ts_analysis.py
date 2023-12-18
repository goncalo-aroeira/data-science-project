from pandas import read_csv, DataFrame, Series, Index, Period
from matplotlib.pyplot import figure, show, savefig, tight_layout, subplots, plot, legend
from dslabs_functions import plot_line_chart, HEIGHT, set_chart_labels, plot_multiline_chart, plot_components, ts_aggregation_by
from numpy import array
from matplotlib.figure import Figure

#***********************************************************************************
#*                                   EX 1
#*                                Granularity                                      *
#***********************************************************************************

def symbolic_variables_granularity(series: Series, file_tag: str, target: str):
    grans: list[str] = ["min", "H","D", "W", "M"]
    for i in range(len(grans)):
        ss: Series = ts_aggregation_by(series, grans[i])
        figure(figsize=(3 * HEIGHT, HEIGHT / 2))
        plot_line_chart(
            ss.index.to_list(),
            ss.to_list(),
            xlabel="days",
            ylabel=target,
            title=f"{grans[i]} mean for {target}",
        )
        tight_layout()
        savefig(f"images/ts_analysis/{file_tag}_granularity_{grans[i]}.png", bbox_inches='tight')

#***********************************************************************************
#*                                   EX 2
#*                               Distribution                                      *
#***********************************************************************************

#****************+++***  Boxplots for individual  vars  ********+++************

def boxplots_individual_num_vars(series: Series, file_tag: str, target:str):
    grans: list[str] = ["min", "H","D", "W"]

    # boxplot 1 by 1
    # for i in grans:
    #     ss: Series = ts_aggregation_by(series, i)
    #     fig = figure(figsize=(2 * HEIGHT, HEIGHT))
    #     fig.suptitle("distribution for "+ i, fontsize=14, fontweight='bold')
    #     boxplot(ss)
    #     savefig(f"images/ts_analysis/{file_tag}_boxplot_{i}.png", bbox_inches='tight')

    fig: Figure
    axs: array
    ss_mins, ss_hours, ss_daily, ss_weekly = ts_aggregation_by(series, grans[0]),ts_aggregation_by(series, grans[1]), ts_aggregation_by(series, grans[2]),ts_aggregation_by(series, grans[3])
    fig, axs = subplots(2, 4, figsize=(2 * HEIGHT, HEIGHT))
    set_chart_labels(axs[0, 0], title="MINUTES")
    axs[0, 0].boxplot(ss_mins)
    set_chart_labels(axs[0, 1], title="HOURLY")
    axs[0, 1].boxplot(ss_hours)
    set_chart_labels(axs[0, 2], title="DAILY")
    axs[0, 2].boxplot(ss_daily)
    set_chart_labels(axs[0, 3], title="WEEKLY")
    axs[0, 3].boxplot(ss_weekly)

    axs[1, 0].grid(False)
    axs[1, 0].set_axis_off()
    axs[1, 0].text(0.2, 0, str(ss_mins.describe()), fontsize="small")

    axs[1, 1].grid(False)
    axs[1, 1].set_axis_off()
    axs[1, 1].text(0.2, 0, str(ss_hours.describe()), fontsize="small")

    axs[1, 2].grid(False)
    axs[1, 2].set_axis_off()
    axs[1, 2].text(0.2, 0, str(ss_daily.describe()), fontsize="small")

    axs[1, 3].grid(False)
    axs[1, 3].set_axis_off()
    axs[1, 3].text(0.2, 0, str(ss_weekly.describe()), fontsize="small")
    savefig(f"images/ts_analysis/{file_tag}_boxplot.png", bbox_inches='tight')


def histograms(series: Series, file_tag: str, target:str):
    grans_str: list[str] = ["H","D", "W"]

    ss_hours: Series = ts_aggregation_by(series, gran_level="H", agg_func=sum)
    ss_days: Series = ts_aggregation_by(series, gran_level="D", agg_func=sum)
    ss_weekly: Series = ts_aggregation_by(series, gran_level="W", agg_func=sum)

    grans: list[Series] = [series, ss_hours, ss_days, ss_weekly]
    gran_names: list[str] = ["By Minutes","Hourly", "Daily", "Weekly"]
    fig: Figure
    axs: array
    fig, axs = subplots(1, len(grans), figsize=(len(grans) * HEIGHT, HEIGHT))
    fig.suptitle(f"{file_tag} {target}")
    for i in range(len(grans)):
        set_chart_labels(axs[i], title=f"{gran_names[i]}", xlabel=target, ylabel="Nr records")
        axs[i].hist(grans[i].values)
    savefig(f"images/ts_analysis/{file_tag}_histogram.png", bbox_inches='tight')
    

def get_lagged_series(series: Series, max_lag: int, delta: int = 1):
    lagged_series: dict = {"original": series, "lag 1": series.shift(1)}
    for i in range(delta, max_lag + 1, delta):
        lagged_series[f"lag {i}"] = series.shift(i)
    return lagged_series

def lag(series: Series, file_tag: str, target:str):
    figure(figsize=(3 * HEIGHT, HEIGHT))
    lags = get_lagged_series(series, 20, 10)
    plot_multiline_chart(series.index.to_list(), lags, xlabel="Timestamp", ylabel=target)
    savefig(f"images/ts_analysis/{file_tag}_lag.png", bbox_inches='tight')

from matplotlib.pyplot import setp
from matplotlib.gridspec import GridSpec


def autocorrelation_study(series: Series, max_lag: int, delta: int = 1):
    k: int = int(max_lag / delta)
    fig = figure(figsize=(4 * HEIGHT, 2 * HEIGHT), constrained_layout=True)
    gs = GridSpec(2, k, figure=fig)

    series_values: list = series.tolist()
    for i in range(1, k + 1):
        ax = fig.add_subplot(gs[0, i - 1])
        lag = i * delta
        ax.scatter(series.shift(lag).tolist(), series_values)
        ax.set_xlabel(f"lag {lag}")
        ax.set_ylabel("original")
    ax = fig.add_subplot(gs[1, :])
    ax.acorr(series, maxlags=max_lag)
    ax.set_title("Autocorrelation")
    ax.set_xlabel("Lags")
    savefig(f"images/ts_analysis/{file_tag}_autocorrelation.png", bbox_inches='tight')
    return

def component_study():
    
    filename = "data/forecast_traffic_single.csv"
    file_tag = "fts"
    target = "Total"
    index = "Timestamp"

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
        title=f"{file_tag} by minutes {target}",
        x_label=series.index.name,
        y_label=target,
    )
    show()
    savefig(f"images/ts_analysis/{file_tag}_components_study.png")

def stationary_study(series: Series, file_tag: str, target:str):
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
    savefig(f"images/ts_analysis/{file_tag}_stationarity_study_1.png")

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
    savefig(f"images/ts_analysis/{file_tag}_stationarity_study_2.png")

from statsmodels.tsa.stattools import adfuller


def eval_stationarity(series: Series) -> bool:
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]:.3f}")
    print(f"p-value: {result[1]:.3f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"\t{key}: {value:.3f}")
    return result[1] <= 0.05

#***********************************************************************************



if __name__ == "__main__":
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

    # print("\n",data.shape)
    # print("\n",data.head)
    # print("entering granularity")
    # data['Age'] = data['Age'].astype(str).str.replace('_', '', regex=False).astype(int)
    
    # granularity
    # symbolic_variables_granularity(series, file_tag, target)
    boxplots_individual_num_vars(series, file_tag, target)
    # histograms(series, file_tag, target)
    # lag(series, file_tag, target)
    # autocorrelation_study(series, 10, 1)
    # I have given up peco imensa desculpa
    # component_study()
    # stationary_study(series, file_tag, target)
    # print(f"The series {('is' if eval_stationarity(series) else 'is not')} stationary")
        # result for eval stationary:
        # ADF Statistic: -9.927
        # p-value: 0.000
        # Critical Values:
        #     1%: -3.433
        #     5%: -2.863
        #     10%: -2.567
        # The series is stationary
