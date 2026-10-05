import os
import pandas as pd

from src.logger import logger


class DataIngestion:

    def __init__(self):

        self.data_path = os.path.join(
            "data",
            "Churn_Modelling.csv"
        )

        self.raw_data_path = os.path.join(
            "artifacts",
            "raw_data.csv"
        )

    def initiate_data_ingestion(self):

        logger.info("Data ingestion started")

        print("Starting data ingestion...")

        # Read CSV file
        df = pd.read_csv(self.data_path)

        logger.info("Dataset loaded successfully")

        print("Dataset loaded successfully.")
        print("Rows:", df.shape[0])
        print("Columns:", df.shape[1])

        # Create artifacts folder
        os.makedirs("artifacts", exist_ok=True)

        # Save raw data
        df.to_csv(
            self.raw_data_path,
            index=False
        )

        logger.info("Raw data saved successfully")

        print(
            "Raw data saved at:",
            self.raw_data_path
        )

        return self.raw_data_path


if __name__ == "__main__":

    ingestion = DataIngestion()

    ingestion.initiate_data_ingestion()