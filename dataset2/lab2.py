from pandas import read_csv, DataFrame
from dslabs_functions import get_variable_types
from math import pi, sin, cos
from numpy import nan

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

def encoding(data: DataFrame, file_tag: str):
    variable_types: dict[str, list] = get_variable_types(data)

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
    data.drop(columns=["Customer_ID", "Name", "SSN"])

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

    # Type_of_Loan
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
    data.drop(columns=["Type_of_Loan"])

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
    data["Month_sin"] = list(map(lambda x: sin(2*pi*month_cycle[x]/7) if type(x) == str else nan, data["Month"].values))
    data["Month_sin"] = list(map(lambda x: cos(2*pi*month_cycle[x]/7) if type(x) == str else nan, data["Month"].values))
    data.drop(columns=["Month"])

    # CreditMix
    creditMix_encoding: dict[str, int] = {"Bad": 0, "Standard": 1, "Good": 2}

    # Credit_History_Age
    # TODO

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
        "Payment_of_Min_Amount": payment_of_min_amount_encoding
    }

    data.replace(encoding, inplace=False)

    print(f"the data shape after the encoding: {data.shape}")

#***************************************************************************************************


if __name__ == "__main__":
    filename = "data/class_credit_score.csv"
    file_tag = "Credit_Score"
    target = "Credit_Score"
    data: DataFrame = read_csv(filename, na_values="", index_col="ID")
    
    stroke: DataFrame = read_csv("data/stroke.csv", na_values="")

    data['Age'] = data['Age'].astype(str).str.replace('_', '', regex=False).astype(int)

    # variable encoding step
    encoding(data, file_tag)

    # print(list(data.columns))
    