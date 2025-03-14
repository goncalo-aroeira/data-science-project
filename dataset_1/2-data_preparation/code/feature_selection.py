from pandas import DataFrame, Index, read_csv
from dslabs_functions import (
    select_low_variance_variables,
    study_variance_for_feature_selection,
    apply_feature_selection,
    select_redundant_variables,
    study_redundancy_for_feature_selection,
    HEIGHT, evaluate_approach, plot_multiline_chart
)
from matplotlib.pyplot import figure, show
from sklearn.model_selection import train_test_split
from math import ceil


target = "CovidPos"
file_tag = "CovidPos"

#data_filename: str = "../data/CovidPos_scaled_minmax.csv"
data_filename: str = "../data/CovidPos_scaled_zscore.csv"
data: DataFrame = read_csv(data_filename)
train, test = train_test_split(data, test_size=0.2, random_state=42)


#print("Original variables", train.columns.to_list())
#vars2drop_variance: list[str] = select_low_variance_variables(train, 0.92, target=target)
#print("Variables to drop", vars2drop_variance, len(vars2drop_variance))

print("study variance")
eval_metric = 'recall'

figure(figsize=(2 * HEIGHT, HEIGHT))
study_variance_for_feature_selection(
    train,
    test,
    target=target,
    max_threshold=1.0,
    lag=0.1,
    metric=eval_metric,
    file_tag=file_tag,
)
show()
exit()

#print("Original variables", train.columns.values)
#vars2drop_redundacy: list[str] = select_redundant_variables(train, target=target, min_threshold= 0.5)
#print("Variables to drop", vars2drop_redundacy)

print("study redundancy")
eval_metric = 'recall'

figure(figsize=(2 * HEIGHT, HEIGHT))
study_redundancy_for_feature_selection(
    train,
    test,
    target=target,
    min_threshold=0.3,
    lag=0.1,
    metric=eval_metric,
    file_tag=file_tag,
)
show()

#vars2drop = vars2drop_variance.copy()
#for el in vars2drop_redundacy:
#    if el not in vars2drop:
#        vars2drop += [el]
#print("final vars2drop:  ", vars2drop)

#print("Aplying feature selection")
#train_cp, test_cp = apply_feature_selection(train, test, vars2drop, filename=f"../data/{file_tag}", tag="redundant")
#print(f"Original data: train={train.shape}, test={test.shape}")
#print(f"After redundant FS: train_cp={train_cp.shape}, test_cp={test_cp.shape}")

