from numpy import ndarray
from pandas import read_csv, DataFrame
from matplotlib.figure import Figure
from matplotlib.pyplot import subplots, savefig
from dslabs_functions import HEIGHT, plot_multi_scatters_chart, get_variable_types

filename = "../../2-data_preparation/data/class_pos_covid_original.csv"
file_tag = "CovidPos"
target = "CovidPos"
data: DataFrame = read_csv(filename)
data = data.dropna()

variable_types = get_variable_types(data)
#data = data.head(1000)
vars: list = data.columns.to_list()
print(vars)

done = ['State', 'Sex', 'GeneralHealth', 'PhysicalHealthDays', 'MentalHealthDays', 'LastCheckupTime', 'PhysicalActivities', 'SleepHours', 'RemovedTeeth', 'HadHeartAttack', 'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD', 'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis', 'HadDiabetes', 'DeafOrHardOfHearing', 'BlindOrVisionDifficulty', 'DifficultyConcentrating', 'DifficultyWalking', 'DifficultyDressingBathing', 'DifficultyErrands', 'SmokerStatus', 'ECigaretteUsage', 'ChestScan', 'RaceEthnicityCategory', 'AgeCategory', 'HeightInMeters', 'WeightInKilograms', 'BMI', 'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver', 'TetanusLast10Tdap', 'CovidPos']

if [] != vars:
    n: int = len(vars) - 1
    for i in range(len(vars)):
        var1: str = vars[i]
        print("var1: ", var1)
        if var1 in done:
            continue
        rows = 1
        cols = n - i
        for j in range(i + 1, len(vars)):
            if (var1 in variable_types["binary"] and vars[j] in variable_types["binary"]):
                cols -= 1 
        fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)
        k = 0
        for j in range(i + 1, len(vars)):
            var2: str = vars[j]
            # binary X binary is not interesting
            if not (var1 in variable_types["binary"] and var2 in variable_types["binary"]):
                #plot_multi_scatters_chart(data, var1, var2, target, ax=axs[i, j - 1])
                plot_multi_scatters_chart(data, var1, var2, target, ax=axs[0, k])
                k += 1
        savefig(f"../images/{file_tag}_sparsity_{var1}_per_class.png", bbox_inches="tight")
else:
    print("Sparsity per class: there are no variables.")