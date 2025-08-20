import os
import sys
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Saves a Python object into a file using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Loads a Python object from a file using dill.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models: dict, param: dict):
    """
    Trains and evaluates multiple models with GridSearchCV.
    Returns a dictionary with model names as keys and a dict containing:
        - 'model': best estimator
        - 'train_r2': R² score on training data
        - 'test_r2': R² score on testing data
    """
    try:
        report = {}

        for name, model in models.items():
            print(f"Training {name}...")

            # Get parameter grid for the current model; default to empty dict if not provided
            para = param.get(name, {})

            # Hyperparameter tuning with GridSearchCV
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            # Use the best estimator
            best_model = gs.best_estimator_

            # Predictions after tuning
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # R² Scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            print(f"{name} -> Train R2: {train_model_score:.4f}, Test R2: {test_model_score:.4f}")

            # Save results in report
            report[name] = {
                "model": best_model,
                "train_r2": train_model_score,
                "test_r2": test_model_score
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
