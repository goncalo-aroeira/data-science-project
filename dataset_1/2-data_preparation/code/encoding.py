from pandas import read_csv, DataFrame
from dslabs_functions import get_variable_types, encode_cyclic_variables, dummify

data: DataFrame = read_csv("../../../class_pos_covid.csv", na_values="")
vars: dict[str, list] = get_variable_types(data)

yes_no: dict[str, int] = {"no": 0, "No": 0, "yes": 1, "Yes": 1}
sex_values: dict[str, int] = {"Male": 0, "Female": 1}

state_values: dict[str, int] = {
    #dummify???
    '''
    'Alabama' 'Alaska' 'Arizona' 'Arkansas' 'California' 'Colorado'
    'Connecticut' 'Delaware' 'District of Columbia' 'Florida' 'Georgia'
    'Hawaii' 'Idaho' 'Illinois' 'Indiana' 'Iowa' 'Kansas' 'Kentucky'
    'Louisiana' 'Maine' 'Maryland' 'Massachusetts' 'Michigan' 'Minnesota'
    'Mississippi' 'Missouri' 'Montana' 'Nebraska' 'Nevada' 'New Hampshire'
    'New Jersey' 'New Mexico' 'New York' 'North Carolina' 'North Dakota'
    'Ohio' 'Oklahoma' 'Oregon' 'Pennsylvania' 'Rhode Island' 'South Carolina'
    'South Dakota' 'Tennessee' 'Texas' 'Utah' 'Vermont' 'Virginia'
    'Washington' 'West Virginia' 'Wisconsin' 'Wyoming' 'Guam' 'Puerto Rico'
    'Virgin Islands'
    '''
}

health_values: dict[str, int] = {
    "Poor": 0,
    "Fair": 1,
    "Good": 2,
    "Very Good": 3,
    "Excelent": 4,
}
last_checkup_time_values: dict[str, int] = {
    #dividir em last year e not last year devido ao baixo numero de valores para as categorias de mais de um ano
    "Within past year (anytime less than 12 months ago)": 0,
    "Within past 2 years (1 year but less than 2 years ago)": 1,
    "Within past 5 years (2 years but less than 5 years ago)": 1,
    "5 or more years ago": 1,
}
removed_teeth_values: dict[str, int] = {
    "None of them": 0,
    "1 to 5": 1,
    "6 or more, but not all": 2,
    "All": 3,
}
had_diabetes_values: dict[str, int] = {
    # dividir em yes e no??? implica perda de detalhe
    "No": 0,
    "Yes": 1,
    "No, pre-diabetes or borderline diabetes": 0,
    "Yes, but only during pregnancy (female)": 1,
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
    # taxonomia em que dividi em hispanic e non-hispanic e depois em raças por ordem de parecença
    "Hispanic": 0,
    "White only, Non-Hispanic": 1,
    "Multiracial, Non-Hispanic": 2,
    "Black only, Non-Hispanic": 3,
    "Other race only, Non-Hispanic": 4,
}
age_category_values: dict[str, int] = {
    # apos estudar a granularidade atraves do grfico do data profiling decidi usar o que tinha os intervalos de 10 anos
    'Age 18 to 24': 0,
    'Age 25 to 29': 0,
    'Age 30 to 34': 1,
    'Age 35 to 39': 1,
    'Age 40 to 44': 2,
    'Age 45 to 49': 2,
    'Age 50 to 54': 3,
    'Age 55 to 59': 3,
    'Age 60 to 64': 4,
    'Age 65 to 69': 4,
    'Age 70 to 74': 5,
    'Age 75 to 79': 5,
    'Age 80 or older': 6,
}
tetanus_last_10_tdap_values: dict[str, int] = {
    # dividir em yes e no??? implica perda de detalhe
    '''
    "Yes, received tetanus shot but not sure what type"
    "No, did not receive any tetanus shot in the past 10 years"
    "Yes, received Tdap"
    "Yes, received tetanus shot, but not Tdap"
    '''
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