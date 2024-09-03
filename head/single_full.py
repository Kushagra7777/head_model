import argparse
from pathlib import Path
import os
import random
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

import logger
from estimator_util import get_estimator
import project
from scipy.stats import mode




def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument(
        '--estimator',
        type=str,
        default='random_forest',
        choices=(
            'random_forest',
            'logistic_regression',
            'logistic_regression_sag',
            'logistic_regression_saga',
            'extra_tree',
            'linear_svc',
            'gbdt',
            'mlp-3',
            'mlp-4',
            'knn-2',
            'knn-4',
            'knn-8',
            'knn-16',
            'knn-32',
            'knn-64',
            'knn-128',
            'knn-256',
            'all'  # Add 'all' option to run all estimators
        )
    )
    parser.add_argument('--n_samples', type=int, default=-1)
    parser.add_argument('--n_features', type=int, default=-1)
    args = parser.parse_args()
    return args


def set_random_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def run_estimator(estimator_name, X_train, y_train, X_test, y_test, label_names, result_dir, seed):
    print(f"Running {estimator_name}...")

    # Apply scaling for logistic regression and linear SVC
    if estimator_name in ['logistic_regression', 'linear_svc']:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if estimator_name == 'logistic_regression':
        estimator = LogisticRegression(max_iter=1000, random_state=seed)
    elif estimator_name == 'linear_svc':
        estimator = LinearSVC(max_iter=1000, random_state=seed)
    elif estimator_name.startswith('knn-'):
        n_neighbors = int(estimator_name.split('-')[1])

        # Ensure n_neighbors does not exceed the number of training samples
        if n_neighbors > len(X_train):
            print(f"Reducing n_neighbors to {len(X_train)} due to insufficient training samples.")
            n_neighbors = len(X_train)

        estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif estimator_name == 'gbdt':
        num_classes = len(np.unique(y_train))
        estimator = xgb.XGBClassifier(objective="multi:softmax", num_class=num_classes, random_state=seed)
    else:
        estimator = get_estimator(estimator_name, gpu=-1, n_out=len(label_names), seed=seed)

    estimator.fit(X_train, y_train)

    pred_test = estimator.predict(X_test)

    # Save predictions to file
    np.savetxt(result_dir / f"submission_{estimator_name}.txt", pred_test, fmt="%i")

    # Report precision, recall, and F1 score, setting zero_division=0 for undefined metrics
    report = classification_report(y_test, pred_test, zero_division=0)
    with open(result_dir / f"classification_report_{estimator_name}.txt", "w") as f:
        f.write(report)

    # Store actual vs predicted labels in a CSV
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': pred_test
    })
    results_df.to_csv(result_dir / f"predicted_vs_actual_{estimator_name}.csv", index=False)

    if hasattr(estimator, "predict_proba"):
        prob_test = estimator.predict_proba(X_test)
    elif hasattr(estimator, "decision_function"):
        prob_test = estimator.decision_function(X_test)
    else:
        prob_test = None

    if prob_test is not None:
        np.save(result_dir / f"prob_{estimator_name}.npy", prob_test)




def run_voting_ensemble(estimators, X_train, y_train, X_test, y_test, label_names, result_dir, seed):
    all_predictions = []
    results_df = pd.DataFrame({'Actual': y_test})

    for estimator_name in estimators:
        print(f"Running {estimator_name}...")

        # Apply scaling for logistic regression and linear SVC
        if estimator_name in ['logistic_regression', 'linear_svc']:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        if estimator_name == 'logistic_regression':
            estimator = LogisticRegression(max_iter=1000, random_state=seed)
        elif estimator_name == 'linear_svc':
            estimator = LinearSVC(max_iter=1000, random_state=seed)
        elif estimator_name.startswith('knn-'):
            n_neighbors = int(estimator_name.split('-')[1])
            if n_neighbors > len(X_train):
                n_neighbors = len(X_train)
            estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
        elif estimator_name == 'gbdt':
            num_classes = len(np.unique(y_train))
            estimator = xgb.XGBClassifier(objective="multi:softmax", num_class=num_classes, random_state=seed)
        else:
            estimator = get_estimator(estimator_name, gpu=-1, n_out=len(label_names), seed=seed)

        # Fit the estimator
        estimator.fit(X_train, y_train)

        # Predict on the test set
        pred_test = estimator.predict(X_test)
        all_predictions.append(pred_test)

        # Save predictions in results_df with column named after the model
        results_df[estimator_name] = pred_test

    # Perform majority voting
    all_predictions = np.array(all_predictions)
    final_predictions, _ = mode(all_predictions, axis=0)
    final_predictions = final_predictions.flatten()

    # Add voting prediction to results_df
    results_df['Voting_Prediction'] = final_predictions

    # Calculate correct and wrong counts for each model and voting
    for col in results_df.columns[1:]:  # Skip 'Actual'
        correct_count = np.sum(results_df['Actual'] == results_df[col])
        wrong_count = len(results_df) - correct_count
        results_df.loc['Correct', col] = correct_count
        results_df.loc['Wrong', col] = wrong_count

    # Save the final predicted vs actual comparison to CSV
    results_df.to_csv(result_dir / "predicted_vs_actual_full.csv", index=True)

    # Save the classification report for the ensemble voting
    report = classification_report(y_test, final_predictions, zero_division=0)
    with open(result_dir / "classification_report_full.txt", "w") as f:
        f.write(report)

    print("Voting ensemble completed and results saved.")

def main():
    args = _parse_args()

    logger.setup_logger()
    set_random_seed(args.seed)

    n_samples = args.n_samples
    n_features = args.n_features
    input_dir = Path(args.input_dir)
    result_dir = Path(args.result_dir)

    X_train = pd.read_csv(input_dir / "train" / "feature_vectors.csv", header=None).values.astype(np.float32)
    X_test = pd.read_csv(input_dir / "test" / "feature_vectors.csv", header=None).values.astype(np.float32)
    y_train = pd.read_csv(input_dir / "train" / "labels.txt", header=None).values[:, 0].astype(np.int32)
    y_test = pd.read_csv(input_dir / "test" / "labels.txt", header=None).values[:, 0].astype(np.int32)

    if n_samples >= 0:
        X_train = X_train[:n_samples]
        y_train = y_train[:n_samples]

    if n_features >= 0:
        X_train = X_train[:, :n_features]
        X_test = X_test[:, :n_features]

    label_names = pd.read_csv(input_dir / "label_names.txt", header=None)

    estimators = [
        'random_forest', 'logistic_regression', 'logistic_regression_sag',
        'logistic_regression_saga', 'extra_tree', 'linear_svc', 'gbdt',
        'mlp-3', 'mlp-4', 'knn-2', 'knn-4', 'knn-8', 'knn-16', 'knn-32',
        'knn-64', 'knn-128', 'knn-256'
    ]

    if args.estimator == 'all':
        run_voting_ensemble(estimators, X_train, y_train, X_test, y_test, label_names, result_dir, args.seed)
    else:
        run_estimator(args.estimator, X_train, y_train, X_test, y_test, label_names, result_dir, args.seed)

if __name__ == '__main__':
    main()





# this file is same as single_estimator but also has option to run all estimator on input_dir in a loop.

# python single_full.py --input_dir work_dir/NGS_2 --result_dir work_dir/results_NGS_2_full --seed 42 --estimator all --n_samples -1 --n_features -1
