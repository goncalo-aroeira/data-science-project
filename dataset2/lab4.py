from pandas import read_csv, DataFrame
from dslabs_functions import (read_train_test_from_files, mlp_study, HEIGHT, plot_evaluation_results,
                              plot_multiline_chart, CLASS_EVAL_METRICS, plot_horizontal_bar_chart,
                              random_forests_study)
from matplotlib.pyplot import show, savefig, figure, subplots
from numpy import array, ndarray, argsort, std
from typing import Literal
from sklearn.tree import DecisionTreeClassifier, plot_tree
from subprocess import call
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

#***************************************************************************************************
#                                          Random Forests                                               *
#***************************************************************************************************

def rf(trnX: DataFrame, trnY:DataFrame, tstX: DataFrame, tstY:DataFrame, eval_metric: str):
    figure()
    print("starting rf")
    best_model, params = random_forests_study(
        trnX,
        trnY,
        tstX,
        tstY,
        nr_max_trees=1000,
        lag=250,
        metric=eval_metric,
    )
    print("got best model")
    savefig(f"images/{file_tag}_rf_{eval_metric}_study.png")

    if eval_metric == "accuracy":
        print("overfitting")
        rf_performance(trnX, trnY, tstX, tstY, best_model, params, labels)
        rf_feature_importance(best_model, eval_metric)
        rf_overfitting(trnX, trnY, tstX, tstY, params, eval_metric)

def rf_performance(trnX: DataFrame, trnY:DataFrame, tstX: DataFrame, tstY:DataFrame, best_model, params, labels):
    prd_trn: array = best_model.predict(trnX)
    prd_tst: array = best_model.predict(tstX)
    figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    savefig(f'images/{file_tag}_rf_{params["name"]}_best_{params["metric"]}_eval.png')

def rf_feature_importance(best_model, eval_metric = "accuracy"):
    stdevs: list[float] = list(
        std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)
    )
    importances = best_model.feature_importances_
    indices: list[int] = argsort(importances)[::-1]
    elems: list[str] = []
    imp_values: list[float] = []
    for f in range(len(vars)):
        elems += [vars[indices[f]]]
        imp_values.append(importances[indices[f]])
        print(f"{f+1}. {elems[f]} ({importances[indices[f]]})")

    figure()
    plot_horizontal_bar_chart(
        elems,
        imp_values,
        error=stdevs,
        title="RF variables importance",
        xlabel="importance",
        ylabel="variables",
        percentage=True,
    )
    savefig(f"images/{file_tag}_rf_{eval_metric}_vars_ranking.png")

def rf_overfitting(trnX: DataFrame, trnY:DataFrame, tstX: DataFrame, tstY:DataFrame, params, eval_metric = "accuracy"):
    d_max: int = params["params"][0]
    feat: float = params["params"][1]
    nr_estimators: list[int] = [i for i in range(2, 2501, 500)]

    y_tst_values: list[float] = []
    y_trn_values: list[float] = []
    acc_metric: str = "accuracy"

    for n in nr_estimators:
        clf = RandomForestClassifier(n_estimators=n, max_depth=d_max, max_features=feat)
        clf.fit(trnX, trnY)
        prd_tst_Y: array = clf.predict(tstX)
        prd_trn_Y: array = clf.predict(trnX)
        y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
        y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))

    figure()
    plot_multiline_chart(
        nr_estimators,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"RF overfitting study for d={d_max} and f={feat}",
        xlabel="nr_estimators",
        ylabel=str(eval_metric),
        percentage=True,
    )
    savefig(f"images/{file_tag}_rf_{eval_metric}_overfitting.png")
#***************************************************************************************************
#                                             MLP                                                  *
#***************************************************************************************************

def mlp(trnX: DataFrame, trnY:DataFrame, tstX: DataFrame, tstY:DataFrame, eval_metric: str):
    figure()
    print("starting mlp")
    best_model, params = mlp_study(
        trnX,
        trnY,
        tstX,
        tstY,
        nr_max_iterations=1000,
        lag=250,
        metric=eval_metric,
    )
    print("got best model")
    savefig(f"images/{file_tag}_mlp_{eval_metric}_study.png")
    show()

    if eval_metric == "accuracy":
        print("overfitting")
        mlp_performance(trnX, trnY, tstX, tstY, best_model, params, labels)
        mlp_overfitting(trnX, trnY, tstX, tstY, params, eval_metric)

def mlp_performance(trnX: DataFrame, trnY:DataFrame, tstX: DataFrame, tstY:DataFrame, best_model, params, labels):
    prd_trn: array = best_model.predict(trnX)
    prd_tst: array = best_model.predict(tstX)
    figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    savefig(f'images/{file_tag}_mlp_{params["name"]}_best_{params["metric"]}_eval.png')
    show()

def mlp_overfitting(trnX: DataFrame, trnY:DataFrame, tstX: DataFrame, tstY:DataFrame, params, eval_metric = "accuracy"):
    lr_type: Literal["constant", "invscaling", "adaptive"] = params["params"][0]
    lr: float = params["params"][1]
    nr_iterations: list[int] = [i for i in range(100, 1001, 100)]

    y_tst_values: list[float] = []
    y_trn_values: list[float] = []
    acc_metric = "accuracy"

    warm_start: bool = False
    for n in nr_iterations:
        clf = MLPClassifier(
            warm_start=warm_start,
            learning_rate=lr_type,
            learning_rate_init=lr,
            max_iter=n,
            activation="logistic",
            solver="sgd",
            verbose=False,
        )
        clf.fit(trnX, trnY)
        prd_tst_Y: array = clf.predict(tstX)
        prd_trn_Y: array = clf.predict(trnX)
        y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
        y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))
        warm_start = True

    figure()
    plot_multiline_chart(
        nr_iterations,
        {"Train": y_trn_values, "Test": y_tst_values},
        title=f"MLP overfitting study for lr_type={lr_type} and lr={lr}",
        xlabel="nr_iterations",
        ylabel=str(eval_metric),
        percentage=True,
    )
    savefig(f"images/{file_tag}_mlp_{eval_metric}_overfitting.png")


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


    eval_metrics = ["recall","precision","auc","f1"]
    fig, axs = subplots(nrows=2, ncols=3, figsize=(3*HEIGHT, 2*HEIGHT), squeeze=False)
    fig.suptitle("Decision trees study for different parameters")
    i, j = 0, 0
    for metric in eval_metrics:
        # mlp(trnX, trnY, tstX, tstY, metric)
        rf(trnX, trnY, tstX, tstY, metric)
  
    
    #DT_best_model_performance(trnX, trnY, tstX, tstY, depth=14)
    #DT_overfitting_study(trnX, trnY, tstX, tstY)
    #DT_variable_importance(trnX, trnY, tstX, tstY, depth=14)
        
    
