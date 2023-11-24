from pandas import read_csv, DataFrame
from dslabs_functions import get_variable_types, encode_cyclic_variables, dummify

data: DataFrame = read_csv("../../../class_pos_covid.csv", na_values="")
vars: dict[str, list] = get_variable_types(data)

yes_no: dict[str, int] = {"no": 0, "No": 0, "yes": 1, "Yes": 1}
sex_values: dict[str, int] = {"Male": 0, "Female": 1}

state_values: dict[str, int] = {
}

health_values: dict[str, int] = {
    "Poor": 0,
    "Fair": 1,
    "Good": 2,
    "Very Good": 3,
    "Excelent": 4,
}
last_checkup_time_values: dict[str, int] = {
}
removed_teeth_values: dict[str, int] = {
}
had_diabetes_values: dict[str, int] = {
}
smoker_status_values: dict[str, int] = {
    "Current smoker - now smokes every day": 0,
    "Current smoker - now smokes some days": 1,
    "Former smoker": 2,
    "Never smoked": 3,
    }
e_cigarrete_values: dict[str, int] = {
    "Use them every day": 0,
    "Use them some days": 1,
    "Not at all (right now)": 2,
    "Never used e-cigarettes in my entire life": 3,
}
race_ethnicity_values: dict[str, int] = {
}
age_category_values: dict[str, int] = {
}
tetanus_last_10_tdap_values: dict[str, int] = {
    # diviir em yes e no
}

encoding: dict[str, dict[str, int]] = {
    "State": state_values,
    "Sex": sex_values,
    "GeneralHealth": health_values,
    "LastCheckupTime": last_checkup_time_values,
    "PhysicalActivities": yes_no,
    "RemovedTeeth": removed_teeth_values,
    "HadHeartAttack": yes_no,
    "HadAngina": yes_no,
    "HadStroke": yes_no,
    "HadAsthma": yes_no,
    "HadSkinCancer": yes_no,
    "HadCOPD": yes_no,
    "HadDepressiveDisorder": yes_no,
    "HadKidneyDisease": yes_no,
    "HadArthritis": yes_no,
    "HadDiabetes": had_diabetes_values,
    "DeafOrHardOfHearing": yes_no,
    "BlindOrVisionDifficulty": yes_no,
    "DificultyConcentrating": yes_no,
    "DifficultyWalking": yes_no,
    "DifficultyDressingBathing": yes_no,
    "DifficultyErrands": yes_no,
    "SmokerStatus": smoker_status_values,
    "ECigarreteUsage": e_cigarrete_values,
    "ChestScan": yes_no,
    "RaceEthnicityCategory": race_ethnicity_values,
    "AgeCategory": age_category_values,
    "AlcoholDrinkers": yes_no,
    "HIVTesting": yes_no,
    "FluVaxLast12": yes_no,
    "PneumoVaxEver": yes_no,
    "TetanusLast10Tdap": tetanus_last_10_tdap_values,
    "HighRiskLastYear": yes_no,
    "CovidPos": yes_no,
}
df: DataFrame = data.replace(encoding, inplace=False)
df.head()

for v in vars["symbolic"]:
    print(v, data[v].unique())