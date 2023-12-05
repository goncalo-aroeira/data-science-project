from pandas import read_csv, DataFrame
from dslabs_functions import (read_train_test_from_files, naive_Bayes_study, plot_evaluation_results,
                             knn_study, overfitting_knn, plot_multiline_chart, CLASS_EVAL_METRICS, DELTA_IMPROVE,
                             plot_horizontal_bar_chart, HEIGHT)
from matplotlib.pyplot import show, savefig, figure, subplots
from numpy import array, ndarray, argsort
from typing import Literal
from sklearn.tree import DecisionTreeClassifier, plot_tree
from subprocess import call

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
        study_overfitting_knn(trnX, trnY, tstX, tstY, params, eval_metric)

def performance_knn(trnX: DataFrame, trnY:DataFrame, tstX: DataFrame, tstY:DataFrame, best_model, params, labels):
    prd_trn: array = best_model.predict(trnX)
    prd_tst: array = best_model.predict(tstX)
    figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    savefig(f'images/{file_tag}_knn_{params["name"]}_best_{params["metric"]}_eval.png')

def study_overfitting_knn(trnX: DataFrame, trnY:DataFrame, tstX: DataFrame, tstY:DataFrame, params ,eval_metric = "accuracy"):
    distance: Literal["manhattan", "euclidean", "chebyshev"] = params["params"][1]
    K_MAX = 50
    kvalues: list[int] = [i for i in range(1, K_MAX, 2)]
    y_tst_values, y_trn_values = overfitting_knn(trnX, trnY, tstX, tstY, distance, file_tag = "Credit_Score", k_max = K_MAX)
    figure()
    plot_multiline_chart(
        kvalues,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"KNN overfitting study for {distance}",
        xlabel="K",
        ylabel=str(eval_metric),
        percentage=True,
    )
    savefig(f"images/{file_tag}_knn_overfitting.png")

#***************************************************************************************************

#***************************************************************************************************
#                                        Decision Trees                                            *
#***************************************************************************************************

def decision_trees_study(
        trnX: ndarray, trnY: array, tstX: ndarray, tstY: array, axes: ndarray, i: int, j: int,
        d_max: int=10, lag:int=2, metric='accuracy'
        ) -> tuple:
    criteria: list[Literal['entropy', 'gini']] = ['entropy', 'gini']
    depths: list[int] = [i for i in range(2, d_max+1, lag)]

    best_model: DecisionTreeClassifier | None = None
    best_params: dict = {'name': 'DT', 'metric': metric, 'params': ()}
    best_performance: float = 0.0

    values: dict = {}
    for c in criteria:
        y_tst_values: list[float] = []
        for d in depths:
            clf = DecisionTreeClassifier(max_depth=d, criterion=c, min_impurity_decrease=0)
            clf.fit(trnX, trnY)
            prdY: array = clf.predict(tstX)
            eval: float = CLASS_EVAL_METRICS[metric](tstY, prdY)
            y_tst_values.append(eval)
            if eval - best_performance > DELTA_IMPROVE:
                best_performance = eval
                best_params['params'] = (c, d)
                best_model = clf
            # print(f'DT {c} and d={d}')
        values[c] = y_tst_values
    print(f'DT best with {best_params['params'][0]} and d={best_params['params'][1]} for metric: {metric}')
    figure()
    plot_multiline_chart(
        depths, 
        values,
        ax=axes[i, j],
        title=f'DT Models ({metric})', 
        xlabel='d', 
        ylabel=metric, 
        percentage=True)
    #savefig(f"images/Credit_Score_DT_{metric}_study.png")

    return best_model, best_params

def DT_best_model_performance(trnX, trnY, tstX, tstY, depth):
    best_model = DecisionTreeClassifier(max_depth=depth, criterion="gini", min_impurity_decrease=0)
    best_model.fit(trnX, trnY)
    params = {'name': 'DT', 'metric': "accuracy", 'params': ("gini", depth)}
    prd_trn: array = best_model.predict(trnX)
    prd_tst: array = best_model.predict(tstX)
    figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    savefig(f'images/{file_tag}_{params["name"]}_best_{params["metric"]}_eval.png')
    show()

def DT_overfitting_study(trnX, trnY, tstX, tstY, metric="accuracy"):
    crit: Literal["entropy", "gini"] = "gini"
    d_max = 25
    depths: list[int] = [i for i in range(2, d_max + 1, 1)]
    y_tst_values: list[float] = []
    y_trn_values: list[float] = []
    acc_metric = "accuracy"
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, criterion=crit, min_impurity_decrease=0)
        clf.fit(trnX, trnY)
        prd_tst_Y: array = clf.predict(tstX)
        prd_trn_Y: array = clf.predict(trnX)
        y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
        y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

    figure()
    plot_multiline_chart(
        depths,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"DT overfitting study for {crit}",
        xlabel="max_depth",
        ylabel=str(metric),
        percentage=True,
    )
    savefig(f"images/{file_tag}_DT_overfitting.png")

def DT_variable_importance(trnX, trnY, tstX, tstY, depth):
    best_model = best_model = DecisionTreeClassifier(max_depth=depth, criterion="gini", min_impurity_decrease=0)
    best_model.fit(trnX, trnY)

    tree_filename: str = f"images/{file_tag}_DT_best_tree"
    max_depth2show = 3
    st_labels: list[str] = [str(value) for value in labels]

    figure(figsize=(14, 6))
    plot_tree(
        best_model,
        max_depth=max_depth2show,
        feature_names=vars,
        class_names=st_labels,
        filled=True,
        rounded=True,
        impurity=False,
        precision=2,
    )
    savefig(tree_filename + ".png")

    importances = best_model.feature_importances_
    indices: list[int] = argsort(importances)[::-1]
    elems: list[str] = []
    imp_values: list[float] = []
    for f in range(len(vars)):
        elems += [vars[indices[f]]]
        imp_values += [importances[indices[f]]]
        print(f"{f+1}. {elems[f]} ({importances[indices[f]]})")

    figure(figsize=(10,15))
    plot_horizontal_bar_chart(
        elems,
        imp_values,
        title="Decision Tree variables importance",
        xlabel="importance",
        ylabel="variables",
        percentage=True,
    )
    savefig(f"images/{file_tag}_DT_vars_ranking.png", bbox_inches='tight')

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
    fig, axs = subplots(nrows=2, ncols=3, figsize=(3*HEIGHT, 2*HEIGHT), squeeze=False)
    fig.suptitle("Decision trees study for different parameters")
    i, j = 0, 0
    for metric in eval_metrics:
        # parameters_nb(trnX, trnY, tstX, tstY, metric, labels)

        # need to evaluate diferent k values
        #parameters_knn(trnX, trnY, tstX, tstY, metric, labels)
        decision_trees_study(trnX, trnY, tstX, tstY, axs, j, i, d_max=20, metric=metric)
        i = i+1 if i < 2 else i
        j = j if i == 2 else j+1
    savefig(f"images/Credit_Score_DT_eval.png")
    show()
    
    #DT_best_model_performance(trnX, trnY, tstX, tstY, depth=14)
    #DT_overfitting_study(trnX, trnY, tstX, tstY)
    #DT_variable_importance(trnX, trnY, tstX, tstY, depth=14)
        
    
