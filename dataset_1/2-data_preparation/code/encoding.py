from pandas import read_csv, DataFrame
from dslabs_functions import get_variable_types, mvi_by_filling, evaluate_approach, plot_multibar_chart
from matplotlib.pyplot import figure, show, savefig, close, legend

data: DataFrame = read_csv("../../../class_pos_covid.csv", na_values="")
vars: dict[str, list] = get_variable_types(data)

yes_no: dict[str, int] = {"no": 0, "No": 0, "yes": 1, "Yes": 1}
sex_values: dict[str, int] = {"Male": 0, "Female": 1}

state_values: dict[str, int] = {
    "Alabama": 0, "Alaska": 1, "Arizona": 2, "Arkansas": 3, "California": 4, "Colorado": 5, "Connecticut": 6, "Delaware": 7, 
    "District of Columbia": 8, "Florida": 9, "Georgia": 10, "Hawaii": 11, "Idaho": 12, "Illinois": 13, "Indiana": 14, "Iowa": 15, 
    "Kansas": 16, "Kentucky": 17, "Louisiana": 18, "Maine": 19, "Maryland": 20, "Massachusetts": 21, "Michigan": 22, "Minnesota": 23,
    "Mississippi": 24, "Missouri": 25, "Montana": 26, "Nebraska": 27, "Nevada": 28, "New Hampshire": 29, "New Jersey": 30, "New Mexico": 31, 
    "New York": 32, "North Carolina": 33, "North Dakota": 34, "Ohio": 35, "Oklahoma": 36, "Oregon": 37, "Pennsylvania": 38, "Rhode Island": 39, 
    "South Carolina": 40, "South Dakota": 41, "Tennessee": 42, "Texas": 43, "Utah": 44, "Vermont": 45, "Virginia": 46, "Washington": 47, 
    "West Virginia": 48, "Wisconsin": 49, "Wyoming": 50, "Guam": 51, "Puerto Rico": 52, "Virgin Islands": 53
}

health_values: dict[str, int] = {
    "Poor": 0,
    "Fair": 1,
    "Good": 2,
    "Very good": 3,
    "Excellent": 4,
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
    "All": 2,
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
    "Current smoker - now smokes some days": 0,
    "Former smoker": 1,
    "Never smoked": 2,
    }
e_cigarrete_values: dict[str, int] = {
    "Use them every day": 0,
    "Use them some days": 0,
    "Not at all (right now)": 1,
    "Never used e-cigarettes in my entire life": 2,
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
    "Age 18 to 24": 0,
    "Age 25 to 29": 0,
    "Age 30 to 34": 1,
    "Age 35 to 39": 1,
    "Age 40 to 44": 2,
    "Age 45 to 49": 2,
    "Age 50 to 54": 3,
    "Age 55 to 59": 3,
    "Age 60 to 64": 4,
    "Age 65 to 69": 4,
    "Age 70 to 74": 5,
    "Age 75 to 79": 5,
    "Age 80 or older": 6,
}
tetanus_last_10_tdap_values: dict[str, int] = {
    "Yes, received tetanus shot but not sure what type": 1,
    "No, did not receive any tetanus shot in the past 10 years": 0,
    "Yes, received Tdap": 1,
    "Yes, received tetanus shot, but not Tdap": 1,
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
    "DifficultyConcentrating": yes_no,
    "DifficultyWalking": yes_no,
    "DifficultyDressingBathing": yes_no,
    "DifficultyErrands": yes_no,
    "SmokerStatus": smoker_status_values,
    "ECigaretteUsage": e_cigarrete_values,
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
data.replace(encoding, inplace=True)
data.to_csv("../data/ccs_vars_encoded.csv")


for v in vars["symbolic"]:
    print(v, data[v].unique())
    
    
    
    
############################################# MV Imputation #############################################

file_tag = "CovidPos"
target = "CovidPos"

vars_with_mv: list = []
for var in data.columns:
    if data[var].isna().sum() > 0:
        vars_with_mv += [var]
print(f"variables with missing values: {vars_with_mv}")

# no variable has a considerable amount of missing values therefore we wont drop columns
# remove rows with a lot of missing values (80%) - number of columns = 40
MIN_MV_IN_A_RECORD_RATIO = 0.8
data.dropna(axis=0, thresh=round(data.shape[1] * MIN_MV_IN_A_RECORD_RATIO, 0), inplace=True)

og_symb_vars = get_variable_types(data)["symbolic"]
og_num_vars = get_variable_types(data)["numeric"]

data_filling_frequent = mvi_by_filling(data, "frequent", og_symb_vars, og_num_vars)
data_filling_frequent.to_csv("../data/ccs_mvi_fill_frequent.csv")
data_filling_knn = mvi_by_filling(data, "knn", og_symb_vars, og_num_vars, 3)
data_filling_knn.to_csv("../data/ccs_mvi_fill_knn.csv")


############################################# MV Evaluation #############################################
frequent_fn = "../data/ccs_mvi_fill_frequent.csv"
knn_fn = "../data/ccs_mvi_fill_knn.csv"
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
savefig(f"../images/{file_tag}_mvi_knn_eval.png")
show()