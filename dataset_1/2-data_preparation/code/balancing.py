from pandas import DataFrame, read_csv, Series, concat
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from numpy import ndarray
from matplotlib.pyplot import figure, savefig, show
from dslabs_functions import evaluate_approach, plot_multibar_chart

target = "CovidPos"
data_filename: str = "../data/CovidPos_scaled_zscore.csv"
data: DataFrame = read_csv(data_filename)


def random_train_test_data_split(data: DataFrame) -> list[DataFrame]:
    train, test = train_test_split(data, test_size=0.18, train_size=0.82)
    return [train, test]

def balancing_evaluation(data_filename: str, strategy: str, data_test: DataFrame):
    data: DataFrame = read_csv(data_filename)
    figure()

    testY = data_test[target]
    eval: dict[str, list] = evaluate_approach(data, data_test, target=target, metric="recall")
    plot_multibar_chart(["NB", "KNN"], eval, title=f"Balacing using {strategy} evaluation", percentage=True)
    savefig(f"../images/CovidPos_balancing_{strategy}.png")
    show()
    data_test[target] = testY
    
print("credit score values",data["CovidPos"].unique())

data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)

# data_train = data.head(int(data.shape[0]*0.8))
# data_test = data.tail(int(data.shape[0]*0.2))
# print("training credit score",data_train["Credit_Score"].unique())
# Approach 1 - undersampling
target_count: Series = data_train[target].value_counts()
positive_class = target_count.idxmin() # 0.0
negative_class = target_count.idxmax() # 1.0
data_positives: Series = data_train[data_train[target] == positive_class]
data_negatives: Series = data_train[data_train[target] == negative_class]

df_neg_sample: DataFrame = DataFrame(data_negatives.sample(len(data_positives)))
df_under: DataFrame = concat([data_positives, df_neg_sample], axis=0)
df_under.to_csv("../data/CovidPos_bal_undersamp.csv", index=False)

print("Undersampling results")
print("Minority class=", positive_class, ":", len(data_positives))
print("Majority class=", negative_class, ":", len(df_neg_sample))
print("Proportion:", round(len(data_positives) / len(df_neg_sample), 2), ": 1")

# Aproach 2 - oversampling
df_pos_sample: DataFrame = DataFrame(data_positives.sample(len(data_negatives), replace=True))
df_over: DataFrame = concat([df_pos_sample, data_negatives], axis=0)
df_over.to_csv(f"../data/CovidPos_bal_over.csv", index=False)

print("Oversampling results")
print("Minority class=", positive_class, ":", len(df_pos_sample))
print("Majority class=", negative_class, ":", len(data_negatives))
print("Proportion:", round(len(df_pos_sample) / len(data_negatives), 2), ": 1")

# Approach 3 - SMOTE
RANDOM_STATE = 42

smote: SMOTE = SMOTE(sampling_strategy="minority", random_state=RANDOM_STATE)
y = data_train.pop(target).values
X: ndarray = data_train.values
smote_X, smote_y = smote.fit_resample(X, y)
df_smote: DataFrame = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
df_smote.columns = list(data_train.columns) + [target]
df_smote.to_csv("../data/CovidPos_bal_SMOTE.csv", index=False)

smote_target_count: Series = Series(smote_y).value_counts()
print("SMOTE results")
print("Minority class=", positive_class, ":", smote_target_count[positive_class])
print("Majority class=", negative_class, ":", smote_target_count[negative_class])
print("Proportion:",round(smote_target_count[positive_class] / smote_target_count[negative_class], 2),": 1",)
print(df_smote.shape)

print("start evaluation 1")
balancing_evaluation("../data/CovidPos_bal_undersamp.csv", "undersampling", data_test)
print("start evaluation 2")
balancing_evaluation("../data/CovidPos_bal_over.csv", "oversampling", data_test)
print("start evaluation 3")
balancing_evaluation("../data/CovidPos_bal_SMOTE.csv", "SMOTE", data_test)


# valores para
# Undersampling = 507
# Oversampling = 502
# SMOTE = 505
