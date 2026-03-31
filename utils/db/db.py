import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import Any, Dict, Optional

import pandas as pd

from utils.db.sql import reformat_sql

logger = logging.getLogger(__name__)


class DB(ABC):
    """
    DB interface
    """

    def __init__(self, staging: bool = False):
        self.staging = staging
        self.environment = "staging" if self.staging else "production"

    @abstractmethod
    def connect(self):
        raise NotImplementedError

    @abstractmethod
    def fetch_columns(self, table_name: str) -> list:
        raise NotImplementedError

    def execute_query(self, sql: str) -> None:
        connection = self.connect()
        cursor = connection.cursor()
        try:
            sql = reformat_sql(sql)
            cursor.execute(sql)
            connection.commit()
        except Exception as e:
            logger.error(e)
            connection.rollback()
            raise e
        finally:
            cursor.close()
            connection.close()

    def count_records(self, table_name: str) -> int:
        connection = self.connect()
        cursor = connection.cursor()
        try:
            cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
            count = cursor.fetchone()[0]
            return count
        except Exception as e:
            logger.error(f"Error in counting records: {e}")
            raise e
        finally:
            cursor.close()
            connection.close()

    def read_to_dataframe(self, sql: str, *, coerce_float: bool = False) -> pd.DataFrame:
        """
        Read data from a database to a pandas dataframe
        Parameters
        ----------
        sql: sql query
        coerce_float: if True, automatically convert decimal.Decimal columns to float

        Returns
        -------

        """
        connection = self.connect()
        try:
            logger.debug(sql)
            return pd.read_sql(sql, connection, coerce_float=coerce_float)
        except Exception as e:
            logger.error(f"Error in reading data to a dataframe: {e}")
            raise e
        finally:
            connection.close()

    def fetch_table_in_chunks(self, table_name: str, order_by: str | list[str], chunk_size: int = 1000) -> pd.DataFrame:
        """
        Fetch table in chunks by ordering the table by a specific column
        Args:
            table_name: name of the table
            chunk_size: number of rows to fetch in each chunk
            order_by: columns to order by

        Returns:

        """
        connection = self.connect()
        cursor = connection.cursor()
        offset = 0
        df_list = []
        if isinstance(order_by, list):
            order_by = ', '.join(order_by)
        while True:
            sql = f'SELECT * FROM {table_name} ORDER BY {order_by} OFFSET {offset} LIMIT {chunk_size}'
            logger.debug(sql)
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame.from_records(cursor.fetchall(), columns=columns)
            df_list.append(df)
            if df.shape[0] < chunk_size:
                break
            offset += chunk_size

        cursor.close()
        connection.close()
        return pd.concat(df_list, ignore_index=True, axis=0)

    # -- UPDATE --
    def update_fields(self, table_name: str, fields: Dict[str, Any], cond: Optional[Dict[str, Any]] = None):
        if not fields:
            return
        sets = ", ".join([f"{k} = {self._format_value(v)}" for k, v in fields.items()])
        sql = f"UPDATE {table_name} SET {sets}"
        if cond:
            cond = " AND ".join([f"{k} = {self._format_value(v)}" for k, v in cond.items()])
            sql += f" WHERE {cond}"
        logger.debug(sql)
        logger.info(f"Updating {table_name} ({self.environment})")
        self.execute_query(sql)

    def single_insert(self, data: dict[str, Any], table_name: str) -> int:
        """
        Insert a single row into a table
        Args:
            data:
            table_name:

        Returns: id of the inserted row

        """
        connection = self.connect()
        cursor = connection.cursor()
        cols = list(data.keys())
        placeholders = ", ".join(["%s"] * len(cols))
        params = [self._coerce_param(data[c]) for c in cols]
        sql = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES ({placeholders})"
        logger.debug(sql)
        logger.info(f"Single inserting into {table_name} ({self.environment})")
        try:
            cursor.execute(sql, params)
            connection.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(e)
            connection.rollback()
            raise e
        finally:
            cursor.close()
            connection.close()

    def single_upsert(self, data: dict[str, Any], table_name: str) -> int:
        """Upsert a single row into a table"""
        connection = self.connect()
        cursor = connection.cursor()
        cols = list(data.keys())
        placeholders = ", ".join(["%s"] * len(cols))
        params = [self._coerce_param(data[c]) for c in cols]
        table_lookup_name = table_name.split(".")[-1].strip("`")
        try:
            table_columns = set(self.fetch_columns(table_lookup_name))
        except Exception as e:
            # Keep upsert behavior if metadata lookup fails; id retrieval may not be available.
            logger.warning(f"Error in fetching column name: {e}")
            table_columns = set()
        setters_parts = []
        for col in data.keys():
            if col == "id" and "id" in table_columns:
                # Force MySQL/MariaDB to expose the existing row id via cursor.lastrowid
                setters_parts.append("`id` = LAST_INSERT_ID(`id`)")
            else:
                setters_parts.append(f"`{col}` = VALUES(`{col}`)")

        if "id" in table_columns and "id" not in data:
            # Handle the common upsert case where id is auto-increment and omitted from input
            setters_parts.append("`id` = LAST_INSERT_ID(`id`)")

        setters = ", ".join(setters_parts)
        sql = (
            f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON DUPLICATE KEY UPDATE {setters}"
        )
        logger.debug(sql)
        logger.info(f"Single inserting into {table_name} ({self.environment})")
        try:
            cursor.execute(sql, params)
            connection.commit()
            return int(cursor.lastrowid)
        except Exception as e:
            logger.error(e)
            connection.rollback()
            raise e
        finally:
            cursor.close()
            connection.close()

    def bulk_upsert(self, data: pd.DataFrame | list[dict[str, Any]], table_name: str) -> None:
        """
        Bulk insert dataframe into table (update if exists)
        Args:
            data:
            table_name:

        Returns: None

        """
        if data is None:
            logger.warning(f'No data to upsert into {table_name}')
            return

        # --- Normalize input to (columns, rows) ---
        if isinstance(data, pd.DataFrame):
            if data.shape[0] == 0:
                logger.warning(f'No data to upsert into {table_name}')
                return
            columns = list(map(str, data.columns))
            # Preserve column order, coerce values
            rows = [
                [self._coerce_param(v) for v in rec]
                for rec in data.to_records(index=False)
            ]
        elif isinstance(data, list):
            if len(data) == 0:
                logger.warning(f'No data to upsert into {table_name}')
                return
            if not all(isinstance(d, dict) for d in data):
                raise TypeError("When passing a list, each item must be a dict.")
            # Use keys from the first row to define order; fill missing with None
            columns = list(data[0].keys())
            rows = [
                [self._coerce_param(item.get(col)) for col in columns]
                for item in data
            ]
        else:
            raise TypeError("`data` must be a pandas.DataFrame or a list of dicts.")

        placeholders = "(" + ", ".join(["%s"] * len(columns)) + ")"
        columns_sql = ", ".join(f"`{c}`" for c in columns)  # backticks for safety

        # Build the ON DUPLICATE KEY UPDATE clause for all columns
        # Note: using VALUES(col) works on MariaDB and MySQL < 8.0.20.
        # For MySQL 8.0.20+ the VALUES() reference is deprecated; see docs.
        # If you need the new style, switch each to: `{col}` = NEW.`{col}` and
        # change the INSERT to "... VALUES ... AS NEW ON DUPLICATE KEY UPDATE ..."
        setters = ", ".join(f"`{col}` = VALUES(`{col}`)" for col in columns)

        sql = (
            f"INSERT INTO `{table_name}` ({columns_sql}) VALUES {placeholders} "
            f"ON DUPLICATE KEY UPDATE {setters}"
        )

        logger.info(f"Upserting {len(rows)} rows into {table_name} ({self.environment})")
        connection = self.connect()
        cursor = connection.cursor()
        try:
            cursor.executemany(sql, rows)
            connection.commit()
            logger.info(f"Inserted/updated {cursor.rowcount} rows into {table_name} ({self.environment})")
        except Exception as e:
            logger.error(e)
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def bulk_upsert_in_chunks(self, df: pd.DataFrame, table_name: str, chunk_size: int = 10000):
        """
        Bulk insert dataframe into table (update if exists) in chunks
        Args:
            df:
            table_name:
            chunk_size:

        Returns: None

        """
        if df.shape[0] == 0:
            logger.warning(f'No data to upsert into {table_name}')
            return
        if df.shape[0] < chunk_size:
            self.bulk_upsert(df, table_name)
            return
        for i in range(0, df.shape[0], chunk_size):
            self.bulk_upsert(df.iloc[i:i + chunk_size], table_name)

    def bulk_insert(self, df: pd.DataFrame, table_name: str) -> list | None:
        """
        Bulk insert dataframe into table
        Args:
            df:
            table_name:

        Returns: list of ids if inserted

        """
        if df is None:
            logger.warning(f'No data to insert into {table_name}')
            return None
        if df.shape[0] == 0:
            logger.warning(f'No data to insert into {table_name}')
            return None
        connection = self.connect()
        cursor = connection.cursor()
        logger.debug(f'Setting up inserting {df.shape[0]} rows into {table_name} ({self.environment})')
        columns = df.columns
        values = df.values
        values = ', '.join([
            '(' + ', '.join(
                [self._format_value(x) for x in y]) + ')' for y in
            values
        ])

        # Form the insert query
        insert_query = f"""insert into {table_name}  ({', '.join(columns)}) values {values}"""

        try:
            # Execute the bulk insert
            cursor.execute(insert_query)
            connection.commit()

            # Retrieve the last inserted ID
            cursor.execute("SELECT LAST_INSERT_ID()")
            last_inserted_id = cursor.fetchone()[0]

            # Generate the list of inserted IDs
            inserted_ids = list(range(last_inserted_id - len(df) + 1, last_inserted_id + 1))
            return inserted_ids
        except Exception as e:
            logger.error(e)
            connection.rollback()
            return None
        finally:
            # Close the connection
            cursor.close()
            connection.close()

    def update_column(self, table_name: str, column_name: str, column_values: list, conditions: list) -> None:
        """
        Update a column in a table given column values and conditions
        :param table_name:
        :param column_name:
        :param column_values: list of values to update
        :param conditions: list of conditions (normally with an identifier)
        :return: None
        """
        if len(column_values) != len(conditions):
            raise ValueError("column_values and conditions must have the same length")
        connection = self.connect()
        cursor = connection.cursor()
        query = f"""
        UPDATE {table_name}
        SET {column_name} = CASE
        {' '.join([f'WHEN {condition} THEN {value}' for condition, value in zip(conditions, column_values)])}
        ELSE {column_name}
        END
        """
        try:
            cursor.execute(query)
            logger.info(f'Updated {cursor.rowcount} rows in {table_name} ({self.environment})')
            connection.commit()
        except Exception as e:
            logger.error(e)
            connection.rollback()
        finally:
            cursor.close()
            connection.close()

    def write_dataframe_to_table(self, df: pd.DataFrame, table_name: str, if_exists: str = 'replace') -> None:
        """
        Write a DataFrame to a table in the database
        Args:
            df: DataFrame to write
            table_name: name of the table
            if_exists: 'fail', 'replace', or 'append'

        Returns: None
        """
        if if_exists not in ["fail", "replace", "append"]:
            raise ValueError("if_exists must be 'fail', 'replace', or 'append'.")

        # Connect to the database
        connection = self.connect()
        cursor = connection.cursor()

        # Prepare the DataFrame columns and types
        columns = df.columns.tolist()
        data_types = {
            "int64": "BIGINT",
            "float64": "DOUBLE",
            "object": "TEXT",
            "datetime64[ns]": "DATETIME",
        }
        column_types = [data_types[str(df[col].dtypes)] for col in columns]

        # Generate the CREATE TABLE query
        create_table_query = f"CREATE TABLE {table_name} ("
        create_table_query += ", ".join(f"{col} {col_type}" for col, col_type in zip(columns, column_types))
        create_table_query += ");"
        try:
            # Handle the `if_exists` logic
            if if_exists == "fail":
                cursor.execute(f"SHOW TABLES LIKE '{table_name}';")
                if cursor.fetchone():
                    raise ValueError(f"Table '{table_name}' already exists.")
            elif if_exists == "replace":
                cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
            elif if_exists == "append":
                cursor.execute(f"SHOW TABLES LIKE '{table_name}';")
                if not cursor.fetchone():
                    raise ValueError(f"Table '{table_name}' does not exist.")
                else:
                    self.bulk_upsert(df, table_name)
                    return

            # Create the table if replacing or if it doesn't exist
            if if_exists in ["replace", "fail"]:
                cursor.execute(create_table_query)

            logger.debug(f'Setting up inserting {df.shape[0]} rows into {table_name} ({self.environment})')
            columns = df.columns
            values = df.values
            values = ', '.join([
                '(' + ', '.join(
                    [self._format_value(x) for x in y]) + ')' for y in
                values
            ])

            # Form the insert query
            insert_query = f"""insert into {table_name}  ({', '.join(columns)}) values {values}"""
            # Retrieve the last inserted ID
            cursor.execute("SELECT LAST_INSERT_ID()")
            # Execute the bulk insert
            cursor.execute(insert_query)
            # Commit changes and close the connection
            connection.commit()
        except Exception as e:
            logger.error(e)
            connection.rollback()
            cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
            connection.commit()
        finally:
            cursor.close()
            connection.close()

    def drop_table(self, table_name: str) -> None:
        connection = self.connect()
        cursor = connection.cursor()
        logger.info(f'Dropping table {table_name} ({self.environment})')
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        connection.commit()
        cursor.close()
        connection.close()

    def check_if_keys_exist(self, table: str, key_name: str, key_list: list) -> list:
        """
        Check if a list of keys exist in table
        :param table: table name
        :param key_name: name of the key column
        :param key_list: list of keys
        :return: list of missing keys
        :example: check if writing companies to a table is successful. In that case, all the keys
        should be written in the database. And the output of this should be an empty list
        """
        query = f"""
        SELECT {key_name}
        FROM {table}
        WHERE {key_name} IN ({', '.join([f"'{key}'" for key in key_list])})
        """
        df = self.read_to_dataframe(reformat_sql(query))
        found_keys = df[key_name].tolist()
        # Compare found_keys and key_list
        missing_keys = [key for key in key_list if key not in found_keys]
        if len(missing_keys) > 0:
            logger.warning(f"{missing_keys} not found in table {table}")
        return missing_keys

    def get_last_insert_id(self):
        connection = self.connect()
        cursor = connection.cursor()
        cursor.execute("SELECT LAST_INSERT_ID()")
        last_insert_id = cursor.fetchone()[0]
        cursor.close()
        return last_insert_id

    @staticmethod
    def _format_value(val: Any) -> str:
        # Check if the value is None or NaN, return 'NULL'
        if val is pd.NA or pd.isnull(val) or val == 'NULL':
            return 'NULL'

        # Check if the value is of dict or list type (which would be valid JSON)
        if isinstance(val, (dict, list)):
            return f"'{json.dumps(val)}'"

        # For other types (integers, floats, etc.), return the value directly as a string
        return f"'{val}'"

    @staticmethod
    def _coerce_param(val):
        # JSON-encode dict/list
        if isinstance(val, (dict, list)):
            return json.dumps(val, ensure_ascii=False)
        # Convert pandas missing scalars (NA/NaN/NaT) to None
        if val is None or val is pd.NA:
            return None
        try:
            is_na = pd.isna(val)
            # pd.isna(list/array) can return array-like; only treat scalar True as NA.
            if isinstance(is_na, bool) and is_na:
                return None
            if hasattr(is_na, "item"):
                try:
                    if bool(is_na.item()):
                        return None
                except Exception:
                    pass
        except Exception:
            # Some objects are not supported by pd.isna; leave them for later checks.
            pass
        # Datetime go through as-is (driver adapts)
        if isinstance(val, (datetime, date)):
            return val
        # Keep None as None; everything else as-is
        return None if val == 'NULL' else val
