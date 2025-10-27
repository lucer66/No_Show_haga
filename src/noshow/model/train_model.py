import pickle
from pathlib import Path
from typing import Dict, Union

import pandas as pd
from dvclive.live import Live
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import make_scorer, confusion_matrix, precision_recall_curve, PrecisionRecallDisplay, f1_score
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import numpy as np


def train_cv_model(
    featuretable: pd.DataFrame,
    output_path: Union[Path, str],
    #classifier: BaseEstimator,
    param_grid: Dict,
    save_dvc_exp: bool = True,
    **kwargs,
) -> None:
    """Use Cross validation to train a model and save results and parameters to dvclive

    Parameters
    ----------
    featuretable : pd.DataFrame
        The featuretable
    output_path : Union[Path, str]
        Path to the output folder where to store the dvc results
    classifier : BaseEstimator
        The classifier to use
    param_grid : Dict
        The parameter grid to search for the best model
    save_dvc_exp : bool
        If we want to save the experiment in DVC, by default True
    kwargs
        Additional arguments to pass to the dvclive.Live context manager
    """

    categorical_col = ['gender', 'hour', 'weekday']
    for col in categorical_col:
        featuretable[col] = featuretable[col].astype('category')
    
    def specificity_score(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tn / (tn + fp)

    def sensitivity_score(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return tp / (tp + fn)

    spec_score = make_scorer(specificity_score)
    sens_score = make_scorer(sensitivity_score)
    
    featuretable["no_show"] = (
        featuretable["no_show"].replace({"no_show": "1", "show": "0"}).astype(int)
    )

    print(featuretable["no_show"].value_counts())

    X, y = featuretable.drop(columns="no_show"), featuretable["no_show"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, shuffle=True
    )

    train_groups = X_train.index.get_level_values("pseudo_id")

    classifier = XGBClassifier(enable_categorical=True)
    #HistGradientBoostingClassifier(categorical_features=["hour", "weekday", "gender"])
    #LinearSVC()
    #LogisticRegression()


    with Live(
        save_dvc_exp=save_dvc_exp,
        dir=str(Path(output_path) / "dvclive"),
        **kwargs,
    ) as live:
        # Define the final pipeline with preprocessor and random forest classifier

        cv = StratifiedGroupKFold()

        # Train the pipeline on the training data
        grid = GridSearchCV(
            classifier,
            param_grid=param_grid,
            cv=cv,
            scoring={"f1": "f1", "pr_auc": "average_precision","roc_auc": "roc_auc", "precision": "precision", "recall": "recall", "specificity": spec_score, "sensitivity": sens_score},
            verbose=3,
            refit="f1",
            n_jobs=5,
        )

        grid.fit(X_train, y_train, groups=train_groups)

        print(grid.best_params_)


        result = permutation_importance(grid, X_train, y_train, n_repeats=2, random_state=42)
        for i in range(len(result.importances_mean)):
            print(f"Feature {X_train.columns[i]}: {result.importances_mean[i]}")



        if hasattr(grid.best_estimator_, 'predict_proba'):
            y_scores = grid.best_estimator_.predict_proba(X_test)[:, 1].astype(float)  # This ensures you're selecting the positive class
        else:
            y_scores = grid.best_estimator_.decision_function(X_test)


            

        live.log_sklearn_plot("roc", y_test, y_scores)
        #live.log_sklearn_plot("calibration", y_test, y_scores)
        live.log_sklearn_plot("precision_recall", y_test, y_scores)

        precision, recall, _ = precision_recall_curve(y_test, y_scores)

        disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        disp.plot()
        plot_path = Path(output_path) / "plots"
        plot_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path / "pr_curve.png", bbox_inches="tight")
        plt.close()


        y_train_pred = grid.best_estimator_.predict(X_train)
        y_test_pred = grid.best_estimator_.predict(X_test)
        #Print train and test F1
        print("Train F1", f1_score(y_train, y_train_pred))
        print("Test F1", f1_score(y_test, y_test_pred))
        
        
        live.log_param("model_name", str(classifier))
        live.log_params(grid.best_params_)
        live.log_metric("best_score", grid.best_score_)
        live.log_metric(
            "f1", grid.cv_results_["mean_test_f1"][grid.best_index_]
        )
        live.log_metric(
            "mean_roc_auc", grid.cv_results_["mean_test_roc_auc"][grid.best_index_]
        )
        live.log_metric(
            "std_roc_auc", grid.cv_results_["std_test_roc_auc"][grid.best_index_]
        )
        live.log_metric(
            "pr_auc", grid.cv_results_["mean_test_pr_auc"][grid.best_index_]
        )
        live.log_metric(
            "mean_precision", grid.cv_results_["mean_test_precision"][grid.best_index_]
        )
        live.log_metric(
            "mean_recall", grid.cv_results_["mean_test_recall"][grid.best_index_]
        )
        live.log_metric(
            "mean_specificity", grid.cv_results_["mean_test_specificity"][grid.best_index_]
        )
        live.log_metric(
            "mean_sensitivity", grid.cv_results_["mean_test_sensitivity"][grid.best_index_]
        )

        model_path = Path(output_path) / "models" / "no_show_model_cv.pickle"
        with open(model_path, "wb") as f:
            pickle.dump(grid.best_estimator_, f)


if __name__ == "__main__":
    project_folder = Path(__file__).parents[3]

    featuretable = pd.read_parquet(
        project_folder / "data" / "processed" / "featuretable.parquet"
    )

    best_model = train_cv_model(
        featuretable=featuretable,
        output_path=project_folder / "output",
        param_grid={
            "learning_rate": [0.05,],
            "n_estimators": [300, 500],
            "max_depth": [5],
            "scale_pos_weight": [8],
            "colsample_bytree": [0.8, 1]
        },
    )

''' 
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [300, 500],
            "max_depth": [3, 7],
            "scale_pos_weight": [8, 10],
            "colsample_bytree": [0.8, 1]
'''

''' HistGradientBoostingClassifier
    "max_iter": [200, 300, 500],
    "learning_rate": [0.01, 0.05, 0.1],
'''
''' LinearSVC
            "penalty": ['l2', 'l1'],
            "class_weight": ['balanced', None],
            "max_iter": [6000, 7000, 8000],
'''
''' logistic regression
            "penalty": ['l2', 'l1', 'elasticnet'],
            "class_weight": [None, 'balanced'],
            "solver": ['liblinear', 'newton-cholesky'],
'''
