"""Implementation of various utility functions
to help with machine learning plots
"""

import warnings
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from pydotplus import graph_from_dot_data
from scipy import interp
from sklearn.metrics import auc, precision_recall_fscore_support, roc_curve
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import export_graphviz


def graph_tree(
    classifier,
    class_names: List[str],
    feature_names: List[str],
    out_file: str = None,
    tree_name: str = "tree",
    filled: bool = True,
    rounded: bool = True,
):

    dot_data = export_graphviz(
        classifier,
        filled,
        rounded,
        class_names,
        feature_names,
        out_file,
    )
    graph = graph_from_dot_data(dot_data)

    graph.write_png(tree_name + ".png")


def versiontuple(v):
    return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    """Plots the decision regions corresponding to a given classifier
    Params:
    X: design matrix (numpy)
    y: target vector (numpy)
    classifier: any ml classifier

    returns:
    plot of decision region for classification
    """

    # setup marker generator and color map
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution),
    )
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.6,
            c=cmap(idx),
            edgecolor="black",
            marker=markers[idx],
            label=cl,
        )

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple("1.9.0"):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn("Please update to NumPy 1.9.0 or newer")
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(
            X_test[:, 0],
            X_test[:, 1],
            c="",
            alpha=1.0,
            edgecolor="black",
            linewidths=1,
            marker="o",
            s=55,
            label="test set",
        )


def plot_learning_curve(
    estimator=None, X=None, y=None, cv=5, train_sizes=None, scoring=None
):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        n_jobs=1,
        scoring=scoring,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(
        train_sizes,
        train_mean,
        color="blue",
        marker="o",
        markersize=5,
        label="training accuracy",
    )

    plt.fill_between(
        train_sizes,
        train_mean + train_std,
        train_mean - train_std,
        alpha=0.15,
        color="blue",
    )

    plt.plot(
        train_sizes,
        test_mean,
        color="green",
        linestyle="--",
        marker="s",
        markersize=5,
        label="validation accuracy",
    )

    plt.fill_between(
        train_sizes,
        test_mean + test_std,
        test_mean - test_std,
        alpha=0.15,
        color="green",
    )
    plt.xlabel("Number of training samples")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.tight_layout()


def plot_validation_curve(
    estimator=None,
    X=None,
    y=None,
    param_range=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    param_name=None,
    cv=5,
    scoring=None,
):

    train_scores, test_scores = validation_curve(
        estimator=estimator,
        X=X,
        y=y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring=None,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(
        param_range,
        train_mean,
        color="blue",
        marker="o",
        markersize=5,
        label="training accuracy",
    )

    plt.fill_between(
        param_range,
        train_mean + train_std,
        train_mean - train_std,
        alpha=0.15,
        color="blue",
    )

    plt.plot(
        param_range,
        test_mean,
        color="green",
        linestyle="--",
        marker="s",
        markersize=5,
        label="validation accuracy",
    )

    plt.fill_between(
        param_range,
        test_mean + test_std,
        test_mean - test_std,
        alpha=0.15,
        color="green",
    )

    plt.grid()
    plt.xscale("log")
    plt.legend(loc="lower right")
    plt.xlabel("Parameter: {}".format(param_name))
    plt.ylabel("Accuracy")


def class_report(y_true, y_pred, y_score=None, average="micro"):
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Error! y_true shape %s is not the same shape as y_pred %s"
            % (y_true.shape, y_pred.shape)
        )
    lb = LabelBinarizer()

    if len(y_true.shape) == 1:
        lb.fit(y_true)

    # Value counts of predictions
    labels, cnt = np.unique(y_pred, return_counts=True)
    n_classes = len(labels)
    pred_cnt = pd.Series(cnt, index=labels)

    metrics_summary = precision_recall_fscore_support(
        y_true=y_true, y_pred=y_pred, labels=labels
    )

    avg = list(
        precision_recall_fscore_support(
            y_true=y_true, y_pred=y_pred, average="weighted"
        )
    )

    metrics_sum_index = ["precision", "recall", "f1-score", "support"]
    class_report_df = pd.DataFrame(
        list(metrics_summary), index=metrics_sum_index, columns=labels
    )

    support = class_report_df.loc["support"]
    total = support.sum()
    class_report_df["avg / total"] = avg[:-1] + [total]

    class_report_df = class_report_df.T
    class_report_df["pred"] = pred_cnt
    class_report_df["pred"].iloc[-1] = total

    if not (y_score is None):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for label_it, label in enumerate(labels):
            fpr[label], tpr[label], _ = roc_curve(
                (y_true == label).astype(int), y_score[:, label_it]
            )

            roc_auc[label] = auc(fpr[label], tpr[label])

        if average == "micro":
            if n_classes <= 2:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(), y_score[:, 1].ravel()
                )
            else:
                fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                    lb.transform(y_true).ravel(), y_score.ravel()
                )

            roc_auc["avg / total"] = auc(
                fpr["avg / total"], tpr["avg / total"]
            )

        elif average == "macro":
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in labels]))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in labels:
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr

            roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])

        class_report_df["AUC"] = pd.Series(roc_auc)

    return class_report_df


def modelfit(
    alg,
    X,
    y,
    performCV=True,
    printFeatureImportance=True,
    cv_folds=5,
    scoring="roc_auc",
    feature_names=None,
):

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn import metrics
    from sklearn.model_selection import cross_val_score

    alg.fit(X, y)
    # Predict training set:
    dtrain_predictions = alg.predict(X)
    dtrain_predprob = alg.predict_proba(X)[:, 1]

    # Perform cross-validation:
    if performCV:
        cv_score = cross_val_score(alg, X, y, cv=cv_folds, scoring=scoring)

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(y, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(y, dtrain_predprob))

    if performCV:
        print(
            "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g"
            % (
                np.mean(cv_score),
                np.std(cv_score),
                np.min(cv_score),
                np.max(cv_score),
            )
        )

    # Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(
            alg.feature_importances_, feature_names
        ).sort_values(ascending=False)
        feat_imp.plot(kind="bar", title="Feature Importances")
        plt.ylabel("Feature Importance Score")


def plot_crossval(
    clf, X, y, cv=None, param_range=None, scoring="roc_auc", param_name=None
):
    from sklearn.model_selection import cross_validate

    init_params = clf.get_params()
    clf_parameters = (
        "max_depth",  # decision trees
        "min_samples_leaf",
        "n_estimators",  # boosting
        "learning_rate",
        "base_estimator__max_depth",
        "base_estimator__min_samples_leaf",
        "kneighborsclassifier__n_neighbors",  # knn
        "kneighborsclassifier__leaf_size",
    )
    # decision trees,boosting and knn *********************************
    if param_name in clf_parameters:
        params = clf.get_params()
        fit_time = []
        train_score = []
        val_score = []
        for i in param_range:
            params[param_name] = i
            clf.set_params(**params)
            params_fit = cross_validate(
                clf, X, y, scoring=scoring, cv=cv, return_train_score=True
            )
            fit_time.append(params_fit.get("fit_time").mean())
            train_score.append(params_fit.get("train_score").mean())
            val_score.append(params_fit.get("test_score").mean())

        plt.plot(fit_time, train_score, "bo")
        plt.plot(fit_time, val_score, "go")
        plt.ylabel("AUC Score")
        plt.xlabel("time in seconds")
        plt.title("Learning Curve,Decision Trees,{}".format(param_name))
        clf.set_params(**init_params)

    # *********************************************************************
    # SVM
    elif param_name == "svc__C":
        params = clf.get_params()
        fit_time = []
        train_score = []
        val_score = []
        if params.get("svc__kernel") == "rbf":
            for i in param_range:
                params["svc__C"] = i
                clf.set_params(**params)
                params_fit = cross_validate(
                    clf, X, y, scoring=scoring, cv=cv, return_train_score=True
                )
                fit_time.append(params_fit.get("fit_time").mean())
                train_score.append(params_fit.get("train_score").mean())
                val_score.append(params_fit.get("test_score").mean())

            plt.plot(fit_time, train_score, "ro", label="train_auc_score")
            plt.plot(fit_time, val_score, "go", label="validation_auc_score")
            plt.ylabel("AUC Score")
            plt.xlabel("time in seconds")
            plt.title("Learning Curve rbf kernel SVM, C param")
            plt.legend(loc="best")
            clf.set_params(**init_params)
        elif params.get("svc__kernel") == "linear":
            for i in param_range:
                params["svc__C"] = i
                clf.set_params(**params)
                params_fit = cross_validate(
                    clf, X, y, scoring=scoring, cv=cv, return_train_score=True
                )
                fit_time.append(params_fit.get("fit_time").mean())
                train_score.append(params_fit.get("train_score").mean())
                val_score.append(params_fit.get("test_score").mean())

            plt.plot(fit_time, train_score, "ro", label="train_auc_score")
            plt.plot(fit_time, val_score, "go", label="validation_auc_score")
            plt.ylabel("AUC Score")
            plt.xlabel("time in seconds")
            plt.title("Learning Curve Linear Kernel SVM, C param")
            plt.legend(loc="best")
            clf.set_params(**init_params)

    elif param_name == "svc__gamma":
        params = clf.get_params()
        fit_time = []
        train_score = []
        val_score = []
        for i in param_range:
            params["svc__gamma"] = i
            clf.set_params(**params)
            params_fit = cross_validate(
                clf, X, y, scoring=scoring, cv=cv, return_train_score=True
            )
            fit_time.append(params_fit.get("fit_time").mean())
            train_score.append(params_fit.get("train_score").mean())
            val_score.append(params_fit.get("test_score").mean())
        plt.plot(fit_time, train_score, "ro", label="train_auc_score")
        plt.plot(fit_time, val_score, "go", label="validation_auc_score")
        plt.ylabel("AUC Score")
        plt.xlabel("time in seconds")
        plt.title("Learning Curve (SVM rbf kernel,gamma param)")
        plt.legend(loc="best")
        clf.set_params(**init_params)
