from dslabs_functions import read_train_test_from_files, naive_Bayes_study, plot_evaluation_results
from matplotlib.pyplot import show, savefig, figure
from numpy import array, ndarray


file_tag = "CovidPos"
train_filename = "../../2-data_preparation/data/CovidPos_bal_undersamp.csv"
test_filename = "../../2-data_preparation/data/CovidPos_test_redundant.csv"
target = "CovidPos"
eval_metric = "accuracy"


trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(
    train_filename, test_filename, target
)
print(f"Train#={len(trnX)} Test#={len(tstX)}")
print(f"Labels={labels}")

figure()
best_model, params = naive_Bayes_study(trnX, trnY, tstX, tstY, "recall")
savefig(f"../images/{file_tag}_nb_recall_study.png")
show()
#Gaussian is the best model
prd_trn: array = best_model.predict(trnX)
prd_tst: array = best_model.predict(tstX)
figure()
plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
savefig(f'../images/{file_tag}_{params["name"]}_best_{params["metric"]}_eval.png')
show()

figure()
best_model, params = naive_Bayes_study(trnX, trnY, tstX, tstY, eval_metric)
savefig(f"../images/{file_tag}_nb_{eval_metric}_study.png")
show()
