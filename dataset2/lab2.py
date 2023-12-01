from pandas import read_csv, DataFrame, Series, concat
from dslabs_functions import (get_variable_types, mvi_by_filling, evaluate_approach, 
                              plot_multibar_chart, determine_outlier_thresholds_for_var)
from math import pi, sin, cos
from numpy import nan, ndarray
from matplotlib.pyplot import show, savefig, figure, close, subplots
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

#***************************************************************************************************
#                                        Variable Encoding                                         *
#***************************************************************************************************

def get_all_unique_loans(data: DataFrame) -> list:
    res = []
    for loan in data["Type_of_Loan"].dropna().unique().tolist():
        for individual_loan in list(map(lambda x: x.strip(), loan.split(','))):
            if "and " in individual_loan:
                individual_loan = individual_loan.replace("and ", "", 1)
            if individual_loan.replace(" ","_") not in res:
                res += [individual_loan.replace(" ","_")]
    return res

def get_unique_loans_for1entry(full_loan_entry: str) -> list:
    return list(map(lambda x: x.strip().replace("and ", "").replace(" ", "_"), full_loan_entry.split(',')))

def get_all_unique_credit_history_age(data: DataFrame) -> list:
    return data["Credit_History_Age"].dropna().unique().tolist()

def transfrom_CHA_to_int(unique_CHA: str) -> int:
    return int(unique_CHA.split(" ")[0]) + round(int(unique_CHA.split(" ")[3])/12, 2)

def encoding(data: DataFrame, file_tag: str):
    # missing vars study (Ze: nao creio que isto seja necessario)
    print('num records', data["CreditMix"].count())
    print('missing records', data["CreditMix"].isna().sum())
    print('percentage of missing vals in CreditMix', data["CreditMix"].isna().sum()/data["CreditMix"].count())
    #  25% missing values not enough to drop Column

    # answer
    #### Nominal Variables ####
    print("IDs - drop")
    print(f"Number of different Customer_IDs: {data["Customer_ID"].dropna().unique().size} - drop") # 12500         				
    print(f"Number of different Names: {data["Name"].dropna().unique().size} - drop") # 10139
    print(f"Number of different SSN: {data["SSN"].dropna().unique().size} - drop") # 12500
    print('Occupation - Ordinal encoding based on taxonomy')
    print('Type_of_Loan - ????????? boa questão, dummification? Ha 9 tipos diferentes de loans, talvez nao seja ma ideia')
    #### Binary Variables ####					
    print('Binary variables - Ordinal linear encoding, 1-0?')
    print("Credit_Score [Poor, Good] - [0, 1]")
    #### Nominal Variables but Cyclic ####				
    print('Month - Ordinal linear encoding, 1-8?, Maybe better perform cyclic encoding')
    #### Ordinal Variables
    print('CreditMix - Ordinal linear encoding [Good, Stantard, Bad] -> [0,1,2]')
    print('Credit_History_Age - Discretization? só para years. Talvez por tipo 3 anos e 6 meses como 3.5??')
    print('Payment_Behaviour - Ordinal encoding based on taxonomy')
    print("Payment_of_Min_Amount - Ordinal linear encoding [No, NM, Yes] -> [0,1,2]")

    # ID, Customer_ID, Name, SSN - dropping...
    # data.drop(columns=["ID", "Customer_ID", "Name", "SSN"], inplace=True)
    # ID
    # for i in range(data["ID"].count()):
    #     if type(data["ID"][i]) == str:
    #         data["ID"][i] = int(data["ID"][i], 16)
    data["ID"] = list(map(lambda x: int(x, 16) if type(x) == str else nan, data["ID"].values))

    # Customer_ID
    data["Customer_ID"] = list(map(lambda x: int(x.replace("CUS_",""), 16) if type(x) == str else nan, data["Customer_ID"].values))

    #SSN
    data["SSN"] = list(map(lambda x: int(x.replace("-","")) if type(x) == str else nan, data["SSN"].values))

    # Name
    names_encoding: dict[str, int] = {}
    n = 0
    for name in data["Name"].dropna().unique():
        names_encoding[name] = n
        n += 1
    data["Name"] = list(map(lambda x: names_encoding[x] if type(x) == str else nan, data["Name"].values))


    # def name_transform(name):
    #     res = 0
    #     for c in name:
    #         res += ord(c)
    #     return res
    # data["Name"] = list(map(lambda x: name_transform(x) if type(x) == str else nan, data["Name"].values))

    # Occupation
    SERVICES = 0
    STEM = 1
    BUSINESS = 2
    MEDIA = 3
    CREATIVE = 4
    occupation_encoding: dict[str, int] = {
        'Scientist': STEM,
        'Teacher': SERVICES,
        'Engineer': STEM,
        'Entrepreneur': BUSINESS,
        'Developer': STEM,
        'Lawyer': SERVICES, 
        'Media_Manager': MEDIA,
        'Doctor': STEM, 
        'Journalist': MEDIA, 
        'Manager': BUSINESS, 
        'Accountant': SERVICES, 
        'Musician': CREATIVE,
        'Mechanic': STEM, 
        'Writer': CREATIVE,
        'Architect': STEM
    }

    # Type_of_Loan - dummification
    unique_loan_rep: dict[str, list] = {}
    for aux in get_all_unique_loans(data):
        unique_loan_rep[aux] = []
    for full_loan_entry in data["Type_of_Loan"].values:
        if type(full_loan_entry) != str: # missing value
            for entry in unique_loan_rep:
                unique_loan_rep[entry] += [nan]
        else:
            single_loans_for1entry = get_unique_loans_for1entry(full_loan_entry)
            for entry in unique_loan_rep:
                unique_loan_rep[entry] += [single_loans_for1entry.count(entry)]
    for entry in unique_loan_rep:
        data[entry] = unique_loan_rep[entry]
    data.drop(columns=["Type_of_Loan"], inplace=True)

    # Credit_Score
    credit_score_encoding: dict[str, int] = {"Poor": 0, "Good": 1}

    # Month
    month_cycle: dict[str, int] = {
        'January': 0,
        'February': 1, 
        'March': 2,
        'April': 3,
        'May': 4,
        'June': 5,
        'July': 6,
        'August': 7,
        'September': 8,
        'October': 9,
        'November': 10,
        'December': 11
    }
    data["Month_sin"] = list(map(lambda x: round(sin(2*pi*month_cycle[x]/12), 2) if type(x) == str else nan, data["Month"].values))
    data["Month_cos"] = list(map(lambda x: round(cos(2*pi*month_cycle[x]/12), 2) if type(x) == str else nan, data["Month"].values))
    data.drop(columns=["Month"], inplace=True)

    # CreditMix
    creditMix_encoding: dict[str, int] = {"Bad": 0, "Standard": 1, "Good": 2}

    # Credit_History_Age
    credit_history_age_encoding: dict[str, int] = {}
    for unique_CHA in get_all_unique_credit_history_age(data):
        credit_history_age_encoding[unique_CHA] = transfrom_CHA_to_int(unique_CHA)

    # Payment_Behaviour
    payment_behaviour_encoding: dict[str, int] = {
        'Low_spent_Small_value_payments': 0,
        'Low_spent_Medium_value_payments': 1,
        'Low_spent_Large_value_payments': 3,
        'High_spent_Small_value_payments': 2, 
        'High_spent_Medium_value_payments': 4,
        'High_spent_Large_value_payments': 5,
    }

    # Payment_of_Min_Amount
    payment_of_min_amount_encoding: dict[str, int] = {'No': 0, 'NM': 1, 'Yes': 2}

    encoding: dict[str, dict[str, int]] = {
        "Occupation": occupation_encoding,
        "Credit_Score": credit_score_encoding,
        "CreditMix": creditMix_encoding,
        "Payment_Behaviour": payment_behaviour_encoding,
        "Payment_of_Min_Amount": payment_of_min_amount_encoding,
        "Credit_History_Age": credit_history_age_encoding
    }

    data.replace(encoding, inplace=True)

    print(f"the data shape after the encoding: {data.shape}")
    data.to_csv("data/ccs_vars_encoded.csv", index=False)

    print(data.head(20))

def newNoMissing(data: DataFrame, file_tag: str):
    variable_types: dict[str, list] = get_variable_types(data)
    print(variable_types)
    for key in data:
        print('Column Name : ', key)
        print('Column Contents : ', data[key].unique())
        print('Missing Records : ', data[key].isna().sum(), '\n')

#***************************************************************************************************

#***************************************************************************************************
#                                   Missing Values Imputation                                      *
#***************************************************************************************************

def missing_values_imputation(data_filename: str, og_symb_vars: list[str], og_num_vars: list[str], unique_loans: list[str]):
    data: DataFrame = read_csv(data_filename)
    vars_with_mv: list = []
    for var in data.columns:
        if data[var].isna().sum() > 0:
            vars_with_mv += [var]
    print(f"variables with missing values: {vars_with_mv}")

    # no variable has a considerable amount of missing values therefore we wont drop columns
    # remove rows with a lot of missing values (90%) - number of columns = 33
    print("og data", data.shape)
    MIN_MV_IN_A_RECORD_RATIO = 0.9
    data.dropna(axis=0, thresh=round(data.shape[1] * MIN_MV_IN_A_RECORD_RATIO, 0), inplace=True)
    print("dropped data", data.shape)
    print("filling frequent")
    data_filling_frequent = mvi_by_filling(data, "frequent", og_symb_vars + unique_loans, og_num_vars)
    data_filling_frequent.to_csv("data/ccs_mvi_fill_frequent.csv", index=False)
    print("filling knn")
    data_filling_knn = mvi_by_filling(data, "knn", og_symb_vars, og_num_vars, 3)
    data_filling_knn.to_csv("data/ccs_mvi_fill_knn.csv", index=False)

def mvi_evaluation(target: str, file_tag: str):
    frequent_fn = "data/ccs_mvi_fill_frequent.csv"
    knn_fn = "data/ccs_mvi_fill_knn.csv"
    data_frequent_mvi_fill: DataFrame = read_csv(frequent_fn)
    data_knn_mvi_fill: DataFrame = read_csv(knn_fn, index_col="ID") ## WHY INDEX_COL?????

    figure()
    eval: dict[str, list] = evaluate_approach(data_frequent_mvi_fill.head(int(data_frequent_mvi_fill.shape[0]*0.8)), 
                                              data_frequent_mvi_fill.tail(int(data_frequent_mvi_fill.shape[0]*0.2)), 
                                              target=target, metric="recall")
    plot_multibar_chart(["NB", "KNN"], eval, title="MVI Frequence Strat evaluation", percentage=True)
    savefig(f"images/{file_tag}_mvi_freq_eval.png")
    show()
    close()

    figure()
    eval: dict[str, list] = evaluate_approach(data_knn_mvi_fill.head(int(data_knn_mvi_fill.shape[0]*0.8)), 
                                              data_knn_mvi_fill.tail(int(data_knn_mvi_fill.shape[0]*0.2)), 
                                              target=target, metric="recall")
    plot_multibar_chart(["NB", "KNN"], eval, title="MVI KNN Strat evaluation", percentage=True)
    savefig(f"images/{file_tag}_mvi_knn_eval.png")
    show()


#***************************************************************************************************

#***************************************************************************************************
#                                       Outliers Treatment                                         *
#***************************************************************************************************

def outliers_treatment(data_filename: str,  og_num_vars: list[str]):
    data: DataFrame = read_csv(data_filename)

    # Approach 1 - Dropping outliers (std_based = true)
    print(f"Data before dropping outliers: {data.shape}")
    data_outliers_rowDrop: DataFrame = data.copy(deep=True)
    summary5: DataFrame = data.describe()
    print(summary5)
    for var in og_num_vars:
        top_threshold, bottom_threshold = determine_outlier_thresholds_for_var(summary5[var]) #mean
        outliers: Series = data_outliers_rowDrop[(data_outliers_rowDrop[var] > top_threshold) | (data_outliers_rowDrop[var] < bottom_threshold)]
        data_outliers_rowDrop.drop(outliers.index, axis=0, inplace=True)
    data_outliers_rowDrop.to_csv("data/ccs_outliers_rowDrop_stdBased.csv", index=False)
    print(f"Data after dropping std outliers: {data_outliers_rowDrop.shape}")

    # Approach 2 - Dropping outliers (std_based = false)
    print(f"Data before dropping outliers: {data.shape}")
    data_outliers_rowDrop: DataFrame = data.copy(deep=True)
    summary5: DataFrame = data.describe()
    for var in og_num_vars:
        top_threshold, bottom_threshold = determine_outlier_thresholds_for_var(summary5[var], std_based=False) #IQR
        outliers: Series = data_outliers_rowDrop[(data_outliers_rowDrop[var] > top_threshold) | (data_outliers_rowDrop[var] < bottom_threshold)]
        data_outliers_rowDrop.drop(outliers.index, axis=0, inplace=True)
    data_outliers_rowDrop.to_csv("data/ccs_outliers_rowDrop_NotStdBased.csv", index=False)
    print(f"Data after dropping outliers: {data_outliers_rowDrop.shape}")

    # Approach 3 - Replacing Outliers (if > than max or < min) with fixed values - using the median
    data_outliers_rep_fixedMedian: DataFrame = data.copy(deep=True)
    for var in og_num_vars:
        top, bottom = determine_outlier_thresholds_for_var(summary5[var])
        median: float = data_outliers_rep_fixedMedian[var].median()
        data_outliers_rep_fixedMedian[var] = data_outliers_rep_fixedMedian[var].apply(lambda x: median if x > top or x < bottom else x)
    data_outliers_rep_fixedMedian.to_csv("data/ccs_outliers_rep_fixed_median.csv", index=True)

    # Approach 4 - Truncating outliers with the max and min
    data_outliers_trunc: DataFrame = data.copy(deep=True)
    for var in og_num_vars:
        top, bottom = determine_outlier_thresholds_for_var(summary5[var])
        data_outliers_trunc[var] = data_outliers_trunc[var].apply(lambda x: top if x > top else bottom if x < bottom else x)
    data_outliers_trunc.to_csv("data/ccs_outliers_trunc_minmax.csv", index=True)

    # Evaluate the different approaches
    outliers_evaluation(data_filename, "original")
    outliers_evaluation("data/ccs_outliers_rowDrop_stdBased.csv", "rowDrop_stdBased")
    outliers_evaluation("data/ccs_outliers_rowDrop_NotStdBased.csv", "rowDrop_NotStdBased")
    outliers_evaluation("data/ccs_outliers_rep_fixed_median.csv", "rep_fixed_median")
    outliers_evaluation("data/ccs_outliers_trunc_minmax.csv", "truncating_minmax")

def outliers_evaluation(data_filename: str, strategy: str):
    data: DataFrame = read_csv(data_filename)
    figure()
    eval: dict[str, list] = evaluate_approach(data.head(int(data.shape[0]*0.8)),
                                              data.tail(int(data.shape[0]*0.2)),
                                              target=target, metric="recall")
    plot_multibar_chart(["NB", "KNN"], eval, title=f"Outliers Treatment with {strategy} evaluation", percentage=True)
    savefig(f"images/Credit_Score_outliers_treat_{strategy}.png")
    show()


#***************************************************************************************************

#***************************************************************************************************
#                                          Scaling                                               *
#***************************************************************************************************
from pandas import read_csv, DataFrame, Series
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def scaling_treatment(data_filename: str, target: str):
    data: DataFrame = read_csv(data_filename)

    # Approach 1 - Standard scaling (z-score transformation)
    vars: list[str] = data.columns.to_list()
    vars.remove(target)
    target_data: Series = data.pop(target)

    transf: StandardScaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(data)
    df_zscore = DataFrame(transf.transform(data), index=data.index)
    df_zscore.columns = vars
    df_zscore[target] = target_data
    df_zscore.to_csv(f"data/ccs_scaled_zscore.csv", index=False)
    print(f"Data after standard scaling: {df_zscore.head(20)}")
    print(get_variable_types(df_zscore))


    transf: MinMaxScaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(data)
    df_minmax = DataFrame(transf.transform(data), index=data.index)
    df_minmax.columns = vars
    df_minmax[target] = target_data
    df_minmax.to_csv(f"data/ccs_scaled_minmax.csv", index=False)
    print(f"Data after minmax scaling: {df_minmax.head(20)}")
    print(get_variable_types(df_minmax))

    scaling_evaluation(data_filename, "Original", target)
    scaling_evaluation("data/ccs_scaled_zscore.csv", "Z-Score", target)
    scaling_evaluation("data/ccs_scaled_minmax.csv", "MinMax", target)

def scaling_evaluation(data_filename: str, strategy: str, target: str):
    data: DataFrame = read_csv(data_filename)
    figure()
    train, test = random_train_test_data_split(data)
    eval: dict[str, list] = evaluate_approach(train, test,
                                              target=target, metric="recall")
    knn_eval = {}
    for entry in eval:
        knn_eval[entry] = eval[entry][1]
    plot_multibar_chart(["KNN"], knn_eval, title=f"Scaling using {strategy} evaluation", percentage=True)
    savefig(f"images/Credit_Score_scaling_{strategy}.png")
    show()

#***************************************************************************************************

#***************************************************************************************************
#                                          Balancing                                               *
#***************************************************************************************************

def random_train_test_data_split(data: DataFrame) -> list[DataFrame]:
    train, test = train_test_split(data, test_size=0.18, train_size=0.82)
    return [train, test]

def balancing(data: DataFrame, data_test: DataFrame, target: str):
    print("credit score values",data["Credit_Score"].unique())
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
    df_under.to_csv("data/ccs_bal_undersamp.csv", index=False)

    print("Undersampling results")
    print("Minority class=", positive_class, ":", len(data_positives))
    print("Majority class=", negative_class, ":", len(df_neg_sample))
    print("Proportion:", round(len(data_positives) / len(df_neg_sample), 2), ": 1")

    # Aproach 2 - oversampling
    df_pos_sample: DataFrame = DataFrame(data_positives.sample(len(data_negatives), replace=True))
    df_over: DataFrame = concat([df_pos_sample, data_negatives], axis=0)
    df_over.to_csv(f"data/ccs_bal_over.csv", index=False)

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
    df_smote.to_csv("data/ccs_bal_SMOTE.csv", index=False)

    smote_target_count: Series = Series(smote_y).value_counts()
    print("SMOTE results")
    print("Minority class=", positive_class, ":", smote_target_count[positive_class])
    print("Majority class=", negative_class, ":", smote_target_count[negative_class])
    print("Proportion:",round(smote_target_count[positive_class] / smote_target_count[negative_class], 2),": 1",)
    print(df_smote.shape)

    print("start evaluation 1")
    balancing_evaluation("data/ccs_bal_undersamp.csv", "undersampling", data_test)
    print("start evaluation 2")
    balancing_evaluation("data/ccs_bal_over.csv", "oversampling", data_test)
    print("start evaluation 3")
    balancing_evaluation("data/ccs_bal_SMOTE.csv", "SMOTE", data_test)

def balancing_evaluation(data_filename: str, strategy: str, data_test: DataFrame):
    data: DataFrame = read_csv(data_filename)
    figure()

    testY = data_test[target]
    eval: dict[str, list] = evaluate_approach(data, data_test, target=target, metric="recall")
    plot_multibar_chart(["NB", "KNN"], eval, title=f"Balacing using {strategy} evaluation", percentage=True)
    savefig(f"images/Credit_Score_balancing_{strategy}.png")
    show()
    data_test[target] = testY

#***************************************************************************************************


if __name__ == "__main__":
    filename = "data/class_credit_score.csv"
    file_tag = "Credit_Score"
    target = "Credit_Score"
    data: DataFrame = read_csv(filename, na_values="")
    data['Age'] = data['Age'].astype(str).str.replace('_', '', regex=False).astype(int)
    
    stroke: DataFrame = read_csv("data/stroke.csv", na_values="")

    # originally symbolic variables
    og_symb_vars = get_variable_types(data)["symbolic"]
    og_symb_vars.remove("Month")
    og_symb_vars.remove("Type_of_Loan")
    # originally numeric variables
    og_num_vars = get_variable_types(data)["numeric"]
    # originally unique loans
    og_unique_loans = get_all_unique_loans(data)

    # variable encoding step
    #encoding(data, file_tag)
    #newNoMissing(data, file_tag)

    # missing values imputation step
    # missing_values_imputation("data/ccs_vars_encoded.csv", og_symb_vars, og_num_vars, og_unique_loans)
    # print("starting evaluation")
    # mvi_evaluation(target, file_tag)

    # frequent

    # outliers treatment
    print("begin outlier treatment")
    mvi_decided_data = "data/ccs_mvi_fill_frequent.csv"
    # outliers_treatment(mvi_decided_data, og_num_vars)

    # drop row std based

    # Scaling
    outliers_decision = "data/ccs_outliers_rowDrop_stdBased.csv"
    # scaling_treatment(outliers_decision, target)
    # zscore

    # TODO feature engineering

    scaling_decision = "data/ccs_scaled_zscore.csv"
    data: DataFrame = read_csv(scaling_decision)
    data_train, data_test = random_train_test_data_split(data)

    # Balacing
    # balancing(data_train, data_test, target)
    # SMOTE