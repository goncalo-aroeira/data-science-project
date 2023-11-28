from pandas import DataFrame, read_csv, Series, concat
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from numpy import ndarray
from matplotlib.pyplot import figure, savefig, show
from dslabs_functions import evaluate_approach, plot_multibar_chart

target = "CovidPos"
data_filename: str = "../data/CovidPos_outliers_rowDrop_NotStdBased.csv"

def random_train_test_data_split(data: DataFrame) -> list[DataFrame]:
    train, test = train_test_split(data, test_size=0.2, train_size=0.8)
    return [train, test]

def balancing_evaluation(data_filename: str, strategy: str):
    data: DataFrame = read_csv(data_filename)
    data=data.sample(frac=1, random_state=42)
    figure()
    eval: dict[str, list] = evaluate_approach(data.head(int(data.shape[0]*0.8)),
                                              data.tail(int(data.shape[0]*0.2)),
                                              target=target, metric="recall")
    plot_multibar_chart(["NB", "KNN"], eval, title=f"Balacing using {strategy} evaluation", percentage=True)
    savefig(f"../images/COvidPos_balancing_{strategy}.png")
    show()
    

data: DataFrame = read_csv(data_filename)
data_balancing_shuffle: DataFrame = data.sample(frac=1, random_state=42)
data_balancing_shuffle.to_csv("../data/data_balancing_shuffle.csv")
data_train = random_train_test_data_split(data_balancing_shuffle)[0]

# Approach 1 - undersampling
target_count: Series = data_train[target].value_counts()
positive_class = target_count.idxmin()
negative_class = target_count.idxmax()
data_positives: Series = data_train[data_train[target] == positive_class]
data_negatives: Series = data_train[data_train[target] == negative_class]

df_neg_sample: DataFrame = DataFrame(data_negatives.sample(len(data_positives)))
df_under: DataFrame = concat([data_positives, df_neg_sample], axis=0)
df_under.to_csv("../data/CovidPos_bal_undersamp.csv", index=False)

print("Minority class=", positive_class, ":", len(data_positives))
print("Majority class=", negative_class, ":", len(df_neg_sample))
print("Proportion:", round(len(data_positives) / len(df_neg_sample), 2), ": 1")

# Approach 2 - SMOTE
RANDOM_STATE = 42

smote: SMOTE = SMOTE(sampling_strategy="minority", random_state=RANDOM_STATE)
y = data_train.pop(target).values
X: ndarray = data_train.values
smote_X, smote_y = smote.fit_resample(X, y)
df_smote: DataFrame = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
df_smote.columns = list(data_train.columns) + [target]
df_smote.to_csv("../data/CovidPos_bal_SMOTE.csv", index=False)

smote_target_count: Series = Series(smote_y).value_counts()
print("Minority class=", positive_class, ":", smote_target_count[positive_class])
print("Majority class=", negative_class, ":", smote_target_count[negative_class])
print("Proportion:",round(smote_target_count[positive_class] / smote_target_count[negative_class], 2),": 1",)
print(df_smote.shape)

balancing_evaluation("../data/CovidPos_bal_undersamp.csv", "undersampling")
balancing_evaluation("../data/CovidPos_bal_SMOTE.csv", "SMOTE")
