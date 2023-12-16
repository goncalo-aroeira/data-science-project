from pandas import read_csv, DataFrame, Series, Index, Period
from matplotlib.pyplot import figure, show, savefig, tight_layout
from dslabs_functions import plot_line_chart, HEIGHT


#***********************************************************************************
#*                                   EX 1
#*                                Granularity                                      *
#*                          2.1 - Symbolic Variables                               *
#***********************************************************************************
def ts_aggregation_by(
    data: Series | DataFrame,
    gran_level: str = "D",
    agg_func: str = "mean",
) -> Series | DataFrame:
    df: Series | DataFrame = data.copy()
    index: Index[Period] = df.index.to_period(gran_level)
    # print(index)
    df = df.groupby(by=index, dropna=True, sort=True).agg(agg_func)
    df.index.drop_duplicates()
    df.index = df.index.to_timestamp()

    return df


def symbolic_variables_granularity(data: DataFrame, file_tag: str, target: str):
    series: Series = data[target]
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
        savefig(f"images/{file_tag}_{grans[i]}_granularity.png", bbox_inches='tight')

#***********************************************************************************



if __name__ == "__main__":
    filename = "data/forecast_traffic_single.csv"
    file_tag = "fts"
    target = "Total"
    data: DataFrame = read_csv(filename, na_values="", index_col="Timestamp",sep=",", decimal=".", parse_dates=True, infer_datetime_format=True)
    
    stroke: DataFrame = read_csv(filename, na_values="")

    # print("\n",data.shape)
    # print("\n",data.head)
    # print("entering granularity")
    # data['Age'] = data['Age'].astype(str).str.replace('_', '', regex=False).astype(int)
    
    # granularity
    symbolic_variables_granularity(data, file_tag, target)

