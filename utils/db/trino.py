# Connect to trinodb
import os
from dotenv import load_dotenv, find_dotenv

import trino

from utils.config import db_cfg
from utils.db.db import DB


class Trino(DB):
    def __init__(self):
        super().__init__()
        # Read env file
        load_dotenv(find_dotenv())

        # Check if env variables are set
        assert os.getenv("TRINO_USERNAME") is not None
        assert os.getenv("TRINO_PASSWORD") is not None

        config = db_cfg["trino"]
        self.user = os.getenv("TRINO_USERNAME")
        self.password = os.getenv("TRINO_PASSWORD")
        self.host = config["host"]
        self.port = config["port"]
        self.schema = config["schema"]
        self.catalog = config["catalog"]

    def connect(self):
        try:
            trino_connection = trino.dbapi.connect(
                http_scheme="https",
                host=self.host,
                port=self.port,
                user=self.user,
                catalog=self.catalog,
                schema=self.schema,
                auth=trino.auth.BasicAuthentication(
                    self.user, self.password
                ),
            )
            return trino_connection
        except trino.exceptions.TrinoConnectionError as e:
            print(f"Error connecting to Trino: {e}")
            raise e

    def fetch_columns(self, table_name: str) -> list:
        query = f"""SELECT COLUMN_NAME FROM information_schema.columns WHERE TABLE_SCHEMA = '{self.schema}' 
        AND TABLE_NAME = '{table_name}';"""
        connection = self.connect()
        cursor = connection.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        return [row[0] for row in rows]
