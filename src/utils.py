import os
import sys
import dill
from sklearn.metrics import r2_score

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


def evaluate_models(X_train, y_train, X_test, y_test, models: dict):
    """
    Trains and evaluates multiple models.
    Returns a dictionary with model names as keys and R² score on test data as values.
    """
    try:
        report = {}

        for name, model in models.items():
           
            model.fit(X_train, y_train)

         
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

           
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            
            print(f"{name} -> Train R2: {train_model_score:.4f}, Test R2: {test_model_score:.4f}")

            report[name] = test_model_score 

        return report

    except Exception as e:
        raise CustomException(e, sys)
