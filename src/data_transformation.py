import os
import pandas as pd
import joblib

from src.logger import logger

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


class DataTransformation:

    def __init__(self):

        self.raw_data_path = os.path.join(
            "artifacts",
            "raw_data.csv"
        )

        self.preprocessor_path = os.path.join(
            "artifacts",
            "preprocessor.pkl"
        )

    def initiate_data_transformation(self):

        logger.info("Data transformation started")

        print("Starting data transformation...")

        # Load raw data
        df = pd.read_csv(self.raw_data_path)

        logger.info("Raw data loaded successfully")

        # Remove unnecessary columns
        df = df.drop(
            columns=[
                "RowNumber",
                "CustomerId",
                "Surname"
            ]
        )

        logger.info("Unnecessary columns removed")

        # Separate features and target
        X = df.drop(columns=["Exited"])
        y = df["Exited"]

        # Categorical columns
        categorical_columns = [
            "Geography",
            "Gender"
        ]

        # Numerical columns
        numerical_columns = [
            "CreditScore",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary"
        ]

        # Numerical preprocessing
        numerical_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler())
            ]
        )

        # Categorical preprocessing
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )
            ]
        )

        # Combine preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numerical",
                    numerical_pipeline,
                    numerical_columns
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_columns
                )
            ]
        )

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        logger.info("Train-test split completed")

        print("Train data:", X_train.shape)
        print("Test data:", X_test.shape)

        # Fit and transform training data
        X_train_transformed = (
            preprocessor.fit_transform(X_train)
        )

        logger.info("Training data transformation completed")

        # Transform test data
        X_test_transformed = (
            preprocessor.transform(X_test)
        )

        logger.info("Testing data transformation completed")

        # Create artifacts directory
        os.makedirs(
            "artifacts",
            exist_ok=True
        )

        # Save transformed training data
        train_data = pd.DataFrame(
            X_train_transformed
        )

        train_data["Exited"] = (
            y_train.reset_index(drop=True)
        )

        train_data.to_csv(
            "artifacts/train_transformed.csv",
            index=False
        )

        # Save transformed testing data
        test_data = pd.DataFrame(
            X_test_transformed
        )

        test_data["Exited"] = (
            y_test.reset_index(drop=True)
        )

        test_data.to_csv(
            "artifacts/test_transformed.csv",
            index=False
        )

        logger.info(
            "Transformed train and test data saved"
        )

        # Save preprocessor
        joblib.dump(
            preprocessor,
            self.preprocessor_path
        )

        logger.info(
            "Preprocessor saved successfully"
        )

        print("Transformation completed.")

        print(
            "Preprocessor saved:",
            self.preprocessor_path
        )

        return (
            "artifacts/train_transformed.csv",
            "artifacts/test_transformed.csv"
        )


if __name__ == "__main__":

    transformation = DataTransformation()

    transformation.initiate_data_transformation()