from pandas import read_csv, DataFrame
from dslabs_functions import get_variable_types

def encoding(data: DataFrame, file_tag: str):
    variable_types: dict[str, list] = get_variable_types(data)

    # Symbolic variable study
    # out = ["Customer_ID","ID","SSN", "Name"]
    # print(variable_types)
    # for key in variable_types['symbolic']:
    #     if key in out:
    #         continue
    #     print(key, ':' ,data[key].unique().tolist())

    # Credit_Score : ['Good', 'Poor'] - binary
    # Month : ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August']
    # Occupation : ['Scientist', nan, 'Teacher', 'Engineer', 'Entrepreneur', 'Developer', 'Lawyer', 'Media_Manager', 'Doctor', 'Journalist', 'Manager', 'Accountant', 'Musician', 'Mechanic', 'Writer', 'Architect']
    # Payment_of_Min_Amount : ['No', 'NM', 'Yes']
    # Payment_Behaviour : ['High_spent_Small_value_payments', 'Low_spent_Large_value_payments', 'Low_spent_Medium_value_payments', 'Low_spent_Small_value_payments', 'High_spent_Medium_value_payments', nan, 'High_spent_Large_value_payments']
    # CreditMix : [nan, 'Good', 'Standard', 'Bad']
    # Credit_History_Age : ['22 Years and 1 Months',...]
    # Type_of_Loan : ['Auto Loan, Credit-Builder Loan, Personal ...]

    # missing vars study
    print('num records', data["CreditMix"].count())
    print('missing records', data["CreditMix"].isna().sum())
    print('percentage of missing vals in CreditMix',data["CreditMix"].isna().sum()/data["CreditMix"].count())
    #  25% missing values not enough to drop Column

    

if __name__ == "__main__":
    filename = "data/class_credit_score.csv"
    file_tag = "Credit_Score"
    target = "Credit_Score"
    data: DataFrame = read_csv(filename, na_values="", index_col="ID")
    
    stroke: DataFrame = read_csv("data/stroke.csv", na_values="")

    data['Age'] = data['Age'].astype(str).str.replace('_', '', regex=False).astype(int)
    encoding(data, file_tag)
    # print(list(data.columns))
    