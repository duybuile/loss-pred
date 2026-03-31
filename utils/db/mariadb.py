# Connect to MariaDB
import os
from dotenv import load_dotenv, find_dotenv

import mariadb

from utils.config import db_cfg
from utils.db.db import DB


class MariaDB(DB):
    def __init__(self, database=None, staging=False):
        super().__init__(staging=staging)
        # Read env file
        load_dotenv(find_dotenv())

        # Check if env variables are set
        assert os.getenv("MARIADB_USERNAME") is not None
        assert os.getenv("MARIADB_PASSWORD") is not None

        config = db_cfg["mariadb"]
        self.user = os.getenv("MARIADB_USERNAME")
        self.password = os.getenv("MARIADB_PASSWORD")
        self.host = config["host"]
        self.port = config["port"]
        if database is not None:
            self.database = database
        else:
            self.database = config["database"]

        if staging:  # If staging is True, use staging database
            self.host = db_cfg["mariadb_staging"]["host"]
            self.port = db_cfg["mariadb_staging"]["port"]
            self.database = db_cfg["mariadb_staging"]["database"]

    def connect(self):
        try:
            mariadb_connection = mariadb.connect(
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                database=self.database,
            )
            return mariadb_connection
        except mariadb.Error as e:
            print(f"Error connecting to MariaDB: {e}")
            raise e

    def fetch_columns(self, table_name: str) -> list:
        query = f"""SELECT COLUMN_NAME FROM information_schema.columns WHERE TABLE_SCHEMA = '{self.database}' 
        AND TABLE_NAME = '{table_name}';"""
        connection = self.connect()
        cursor = connection.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        cursor.close()
        return [row[0] for row in rows]
