from pandas import read_csv, DataFrame
from dslabs_functions import get_variable_types, mvi_by_filling, evaluate_approach, plot_multibar_chart
from math import pi, sin, cos
from numpy import nan
from matplotlib.pyplot import show, savefig, figure, close

#***************************************************************************************************
#                                        Variable Encoding                                         *
#***************************************************************************************************

def get_all_unique_loans(data: DataFrame) -> list:
    res = []
    for loan in data["Type_of_Loan"].dropna().unique().tolist():
        for individual_loan in list(map(lambda x: x.strip(), loan.split(','))):
            if "and " in individual_loan:
                individual_loan = individual_loan.replace("and ", "", 1)
            if individual_loan not in res:
                res += [individual_loan]
    return res

def get_unique_loans_for1entry(full_loan_entry: str) -> list:
    return list(map(lambda x: x.strip(), full_loan_entry.split(',')))

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
    data.drop(columns=["Customer_ID", "Name", "SSN"], inplace=True)

    # Occupation
    STEM = 0
    BUSINESS = 2
    MEDIA = 2
    SERVICES = 3
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
            unique_loans = get_unique_loans_for1entry(full_loan_entry)
            for entry in unique_loan_rep:
                if entry in unique_loans:
                    unique_loan_rep[entry] += [1]
                else:
                    unique_loan_rep[entry] += [0]
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
        'August': 7
    }
    data["Month_sin"] = list(map(lambda x: round(sin(2*pi*month_cycle[x]/7), 2) if type(x) == str else nan, data["Month"].values))
    data["Month_cos"] = list(map(lambda x: round(cos(2*pi*month_cycle[x]/7), 2) if type(x) == str else nan, data["Month"].values))
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

#***************************************************************************************************

#***************************************************************************************************
#                                   Missing Values Imputation                                      *
#***************************************************************************************************

def missing_values_imputation(data: DataFrame, og_symb_vars: list[str], og_num_vars: list[str]):
    vars_with_mv: list = []
    for var in data.columns:
        if data[var].isna().sum() > 0:
            vars_with_mv += [var]
    print(f"variables with missing values: {vars_with_mv}")

    # no variable has a considerable amount of missing values therefore we wont drop columns
    # remove rows with a lot of missing values (80%) - number of columns = 33
    MIN_MV_IN_A_RECORD_RATIO = 0.8
    data.dropna(axis=0, thresh=round(data.shape[1] * MIN_MV_IN_A_RECORD_RATIO, 0), inplace=True)

    data_filling_frequent = mvi_by_filling(data, "frequent", og_symb_vars, og_num_vars)
    data_filling_frequent.to_csv("data/ccs_mvi_fill_frequent.csv")
    data_filling_knn = mvi_by_filling(data, "knn", og_symb_vars, og_num_vars, 3)
    data_filling_knn.to_csv("data/ccs_mvi_fill_knn.csv")

def mvi_evaluation(target: str, file_tag: str):
    frequent_fn = "data/ccs_mvi_fill_frequent.csv"
    knn_fn = "data/ccs_mvi_fill_knn.csv"
    data_frequent_mvi_fill: DataFrame = read_csv(frequent_fn)
    data_knn_mvi_fill: DataFrame = read_csv(knn_fn, index_col="ID")

    figure()
    eval: dict[str, list] = evaluate_approach(data_frequent_mvi_fill.head(int(data_frequent_mvi_fill.shape[0]*0.8)), 
                                              data_frequent_mvi_fill.tail(int(data_frequent_mvi_fill.shape[0]*0.2)), 
                                              target=target, metric="recall")
    plot_multibar_chart(
        ["NB", "KNN"], eval, title=f"{file_tag} evaluation", percentage=True
    )
    savefig(f"images/{file_tag}_mvi_freq_eval.png")
    show()
    close()

    figure()
    eval: dict[str, list] = evaluate_approach(data_knn_mvi_fill.head(int(data_knn_mvi_fill.shape[0]*0.8)), 
                                              data_knn_mvi_fill.tail(int(data_knn_mvi_fill.shape[0]*0.2)), 
                                              target=target, metric="recall")
    plot_multibar_chart(
        ["NB", "KNN"], eval, title=f"{file_tag} evaluation", percentage=True
    )
    savefig(f"images/{file_tag}_mvi_knn_eval.png")
    show()


#***************************************************************************************************


if __name__ == "__main__":
    filename = "data/class_credit_score.csv"
    file_tag = "Credit_Score"
    target = "Credit_Score"
    data: DataFrame = read_csv(filename, na_values="", index_col="ID")
    data['Age'] = data['Age'].astype(str).str.replace('_', '', regex=False).astype(int)
    
    stroke: DataFrame = read_csv("data/stroke.csv", na_values="")

    # originally symbolic variables
    og_symb_vars = get_variable_types(data)["symbolic"]
    og_symb_vars.remove("Customer_ID")
    og_symb_vars.remove("Name")
    og_symb_vars.remove("SSN")
    og_symb_vars.remove("Month")
    og_symb_vars.remove("Type_of_Loan")
    # originally numeric variables
    og_num_vars = get_variable_types(data)["numeric"]

    # variable encoding step
    #encoding(data, file_tag)
    #data.head(20).to_csv("data_vars_encoded.csv")

    # missing values imputation step
    #print()
    #missing_values_imputation(data, og_symb_vars, og_num_vars)
    mvi_evaluation(target, file_tag)

    # print(list(data.columns))
    