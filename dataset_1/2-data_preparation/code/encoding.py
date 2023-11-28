from pandas import read_csv, DataFrame
from dslabs_functions import get_variable_types, mvi_by_filling, mvi_by_dropping, evaluate_approach, plot_multibar_chart
from matplotlib.pyplot import figure, savefig, close, legend

data: DataFrame = read_csv("../../../class_pos_covid.csv", na_values="")
vars: dict[str, list] = get_variable_types(data)

yes_no: dict[str, int] = {"no": 0, "No": 0, "yes": 1, "Yes": 1}
sex_values: dict[str, int] = {"Male": 0, "Female": 1}

state_values: dict[str, int] = {
    #DC não é um estado, mas pertence à regiao 3
    "Alabama": 3, "Alaska": 4, "Arizona": 4, "Arkansas": 3, "California": 4, "Colorado": 4, "Connecticut": 1, "Delaware": 3, 
    "District of Columbia": 3, "Florida": 3, "Georgia": 3, "Hawaii": 4, "Idaho": 4, "Illinois": 2, "Indiana": 2, "Iowa": 2, 
    "Kansas": 2, "Kentucky": 3, "Louisiana": 3, "Maine": 1, "Maryland": 3, "Massachusetts": 1, "Michigan": 2, "Minnesota": 2,
    "Mississippi": 3, "Missouri": 2, "Montana": 4, "Nebraska": 2, "Nevada": 4, "New Hampshire": 1, "New Jersey": 1, "New Mexico": 4, 
    "New York": 1, "North Carolina": 3, "North Dakota": 2, "Ohio": 2, "Oklahoma": 3, "Oregon": 4, "Pennsylvania": 1, "Rhode Island": 1, 
    "South Carolina": 3, "South Dakota": 2, "Tennessee": 3, "Texas": 3, "Utah": 4, "Vermont": 1, "Virginia": 3, "Washington": 4, 
    "West Virginia": 3, "Wisconsin": 2, "Wyoming": 4, "Guam": 5, "Puerto Rico": 5, "Virgin Islands": 5
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
og_symb_vars = get_variable_types(data)["symbolic"]
og_num_vars = get_variable_types(data)["numeric"]
data.replace(encoding, inplace=True)
data.to_csv("../data/ccs_vars_encoded.csv", index=False)
    
def newNoMissing(data: DataFrame, file_tag: str):
    variable_types: dict[str, list] = get_variable_types(data)
    print(variable_types)
    for key in data:
        print('Column Name : ', key)
        print('Column Contents : ', data[key].unique())
        print('Missing Records : ', data[key].isna().sum(), '\n')
   
    
############################################# MV Imputation #############################################

file_tag = "CovidPos"
target = "CovidPos"

# no variable has a considerable amount of missing values therefore we wont drop columns
# remove rows with a lot of missing values (85%) - number of columns = 40
data=mvi_by_dropping(data, 0.85, 0.90)

print(int(data.shape[0]))
print(int(data.shape[1]))

""" print("frequent")
data_filling_frequent = mvi_by_filling(data, "frequent", og_symb_vars, og_num_vars, 3)
data_filling_frequent.to_csv("../data/ccs_mvi_fill_frequent.csv")
print("knn")
data_filling_knn = mvi_by_filling(data, "knn", og_symb_vars, og_num_vars, 3)
data_filling_knn.to_csv("../data/ccs_mvi_fill_knn.csv") """

############################################# MV Evaluation #############################################
frequent_fn = "../data/ccs_mvi_fill_frequent.csv"
knn_fn = "../data/ccs_mvi_fill_knn.csv"
data_frequent_mvi_fill: DataFrame = read_csv(frequent_fn)
data_knn_mvi_fill: DataFrame = read_csv(knn_fn)

data_frequent_shuffle: DataFrame = data_frequent_mvi_fill.sample(frac=1, random_state=42)
data_frequent_shuffle.to_csv("../data/ccs_mvi_fill_frequent_shuffle.csv")
data_knn_shuffle: DataFrame = data_knn_mvi_fill.sample(frac=1, random_state=42)
data_knn_shuffle.to_csv("../data/ccs_mvi_fill_knn_shuffle.csv")

print("Frequent")

figure()
eval: dict[str, list] = evaluate_approach(data_frequent_shuffle.head(int(data_frequent_shuffle.shape[0]*0.6)), 
                                              data_frequent_shuffle.tail(int(data_frequent_shuffle.shape[0]*0.4)), 
                                              target=target, metric="recall")

plot_multibar_chart(["NB", "KNN"], eval, title=f"{file_tag} evaluation", percentage=True)
savefig(f"../images/{file_tag}_mvi_freq_eval.png")
close()

print("KNN")

figure()
eval: dict[str, list] = evaluate_approach(data_knn_shuffle.head(int(data_knn_shuffle.shape[0]*0.6)), 
                                            data_knn_shuffle.tail(int(data_knn_shuffle.shape[0]*0.4)), 
                                            target=target, metric="recall")

plot_multibar_chart(["NB", "KNN"], eval, title=f"{file_tag} evaluation", percentage=True)
savefig(f"../images/{file_tag}_mvi_knn_eval.png")