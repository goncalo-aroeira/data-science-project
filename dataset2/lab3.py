from pandas import read_csv, DataFrame, Series, concat
from dslabs_functions import (read_train_test_from_files, naive_Bayes_study, plot_evaluation_results,
                             knn_study)
from matplotlib.pyplot import show, savefig, figure, close, subplots
from numpy import array, ndarray

#***************************************************************************************************
#                                         Naive Bayes                                              *
#***************************************************************************************************

def parameters_nb(trnX: DataFrame, trnY:DataFrame, tstX: DataFrame, tstY:DataFrame, eval_metric: str, labels):

    figure()
    best_model, params = naive_Bayes_study(trnX, trnY, tstX, tstY, eval_metric)
    savefig(f"images/{file_tag}_nb_{eval_metric}_study.png")

    # o evaluation de qualquer metric é igual para qualquer dos modelos (Bernoulli ou Gaussian)
    if eval_metric == "accuracy":
        performance_nb(trnX, trnY, tstX, tstY, best_model, params, labels)

def performance_nb(trnX: DataFrame, trnY:DataFrame, tstX: DataFrame, tstY:DataFrame, best_model, params, labels):
    prd_trn: array = best_model.predict(trnX)
    prd_tst: array = best_model.predict(tstX)
    figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    savefig(f'images/{file_tag}_nb_{params["name"]}_best_{params["metric"]}_eval.png')

#***************************************************************************************************
#                                             KNN                                                  *
#***************************************************************************************************

def parameters_knn(trnX: DataFrame, trnY:DataFrame, tstX: DataFrame, tstY:DataFrame, eval_metric: str, labels):
    print("knn", eval_metric)
    figure()
    best_model, params = knn_study(trnX, trnY, tstX, tstY, k_max=25, metric=eval_metric)
    savefig(f'images/{file_tag}_knn_{eval_metric}_study.png')  

    # o evaluation de qualquer metric é igual para qualquer dos modelos (Bernoulli ou Gaussian)
    if eval_metric == "accuracy":
        performance_knn(trnX, trnY, tstX, tstY, best_model, params, labels)

def performance_knn(trnX: DataFrame, trnY:DataFrame, tstX: DataFrame, tstY:DataFrame, best_model, params, labels):
    prd_trn: array = best_model.predict(trnX)
    prd_tst: array = best_model.predict(tstX)
    figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    savefig(f'images/{file_tag}_knn_{params["name"]}_best_{params["metric"]}_eval.png')

#***************************************************************************************************

if __name__ == "__main__":
    train_filename = "data/ccs_bal_SMOTE.csv"
    test_filename = "data/ccs_data_fe_test_res.csv"
    file_tag = "Credit_Score"
    target = "Credit_Score"
    
    trnX: ndarray
    tstX: ndarray
    trnY: array
    tstY: array
    labels: list
    vars: list
    trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(
        train_filename, test_filename, target
    )
    print(f"Train#={len(trnX)} Test#={len(tstX)}")
    print(f"Labels={labels}")


    eval_metrics = ["accuracy","recall","precision","auc","f1"]
    for metric in eval_metrics:
        parameters_nb(trnX, trnY, tstX, tstY, metric, labels)
        parameters_knn(trnX, trnY, tstX, tstY, metric, labels)
        
    
