import sys
import os
import logging
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

# Global compatibility patches for older pickled scikit-learn models
try:
    from sklearn.pipeline import Pipeline
    if not hasattr(Pipeline, 'transform_input'):
        Pipeline.transform_input = None
except Exception:
    pass

try:
    from sklearn.impute import SimpleImputer
    if not hasattr(SimpleImputer, '_fill_dtype'):
        def _get_fill_dtype(self):
            stats = getattr(self, "statistics_", None)
            return stats.dtype if stats is not None else object
        SimpleImputer._fill_dtype = property(_get_fill_dtype)
except Exception:
    pass


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found at {preprocessor_path}")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            expected_columns = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course',
                'reading score',
                'writing score'
            ]

            if set(features.columns) != set(expected_columns):
                raise ValueError(f"Input features do not match expected columns.\nExpected: {expected_columns}\nGot: {list(features.columns)}")

            # Ensure correct column order
            features = features[expected_columns]

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            logging.debug("Error during prediction:\n%s", e)
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: float,
                 writing_score: float):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score],
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
