from numpy import array, ndarray, std, argsort
from matplotlib.pyplot import subplots, figure, savefig, show
from sklearn.ensemble import GradientBoostingClassifier
from dslabs_functions import (
    CLASS_EVAL_METRICS,
    DELTA_IMPROVE,
    read_train_test_from_files,
)
from dslabs_functions import HEIGHT, plot_evaluation_results, plot_multiline_chart, gradient_boosting_study, plot_horizontal_bar_chart



file_tag = "CovidPos"
train_filename = "../data/CovidPos_bal_undersamp.csv"
test_filename = "../data/CovidPos_test_redundant.csv"
target = "CovidPos"
eval_metrics = ["precision", "recall", "f1", "auc"]

print("GB")

for eval_metric in eval_metrics:
    
    trnX, tstX, trnY, tstY, labels, vars = read_train_test_from_files(
        train_filename, test_filename, target
    )
    print(f"Train#={len(trnX)} Test#={len(tstX)}")
    print(f"Labels={labels}")
    
    print(f"Metric={eval_metric}")
    figure()
    best_model, params = gradient_boosting_study(
        trnX,
        trnY,
        tstX,
        tstY,
        nr_max_trees=1000,
        lag=250,
        metric=eval_metric,
    )
    savefig(f"../images/{file_tag}_gb_{eval_metric}_study.png")
    show()


    prd_trn: array = best_model.predict(trnX)
    prd_tst: array = best_model.predict(tstX)
    figure()
    plot_evaluation_results(params, trnY, prd_trn, tstY, prd_tst, labels)
    savefig(f'../images/{file_tag}_gb_{params["name"]}_best_{params["metric"]}_eval.png')
    show()



    trees_importances: list[float] = []
    for lst_trees in best_model.estimators_:
        for tree in lst_trees:
            trees_importances.append(tree.feature_importances_)

    stdevs: list[float] = list(std(trees_importances, axis=0))
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
        title="GB variables importance",
        xlabel="importance",
        ylabel="variables",
        percentage=True,
    )
    savefig(f"../images/{file_tag}_gb_{eval_metric}_vars_ranking.png")



'''
d_max: int = params["params"][0]
lr: float = params["params"][1]
nr_estimators: list[int] = [i for i in range(2, 2501, 500)]

y_tst_values: list[float] = []
y_trn_values: list[float] = []
acc_metric: str = "accuracy"

for n in nr_estimators:
    clf = GradientBoostingClassifier(n_estimators=n, max_depth=d_max, learning_rate=lr)
    clf.fit(trnX, trnY)
    prd_tst_Y: array = clf.predict(tstX)
    prd_trn_Y: array = clf.predict(trnX)
    y_tst_values.append(CLASS_EVAL_METRICS[acc_metric](tstY, prd_tst_Y))
    y_trn_values.append(CLASS_EVAL_METRICS[acc_metric](trnY, prd_trn_Y))
        
        
figure()
plot_multiline_chart(
    nr_estimators,
    {"Train": y_trn_values, "Test": y_tst_values},
    title=f"GB overfitting study for d={d_max} and lr={lr}",
    xlabel="nr_estimators",
    ylabel=str(eval_metric),
    percentage=True,
)
savefig(f"../images/{file_tag}_gb_{eval_metric}_overfitting.png")
'''