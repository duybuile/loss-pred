# Reformat SQL commands, read SQL files
import re


def reformat_sql(sql: str) -> str:
    """
    Reformat SQL string by removing comments, trailing whitespaces, line skipping,
    tabs, and trailing semicolons.

    Args:
        sql (str): SQL string to be reformatted.

    Returns:
        str: Reformatted SQL string.
    """
    # Remove lines of comments
    # Regex pattern matches /*...*/ and --...\n comments
    sql = re.sub(r"/\*.*?\*/|--.*?\n", "", sql)

    # Remove trailing whitespaces
    sql = sql.strip()

    # Replace newline characters with spaces
    sql = sql.replace("\n", " ")

    # Remove tabs
    sql = sql.replace("\t", "")

    # If it ends with a semicolon, remove it
    sql = sql.rstrip(";")

    return sql


def read_sql_file(sql_file: str) -> str:
    """
    Read SQL file.

    Reads the content of the SQL file specified by the 'sql_file' parameter.

    Args:
        sql_file (str): The path to the SQL file.

    Raises:
        Exception: If the provided file is not a SQL file.

    Returns:
        str: The content of the SQL file.
    """
    # Check if the file is a SQL file
    if not sql_file.endswith(".sql"):
        # Raise an exception if the provided file is not a SQL file
        raise Exception(f"File {sql_file} is not a SQL file")

    # Open the SQL file in read mode
    with open(sql_file, "r") as file:
        # Read the content of the file
        sql = file.read()

    # Return the content of the SQL file
    return sql


def read_multiple_commands_sql_file(sql_file: str) -> list:
    """
    Read multiple commands from an SQL file.

    Reads the content of the SQL file specified by the 'sql_file' parameter and
    splits it into individual SQL commands.

    Args:
        sql_file (str): The path to the SQL file.

    Returns:
        list: A list containing individual SQL commands.
    """
    # Read the SQL file and split it into individual commands
    # Check if the file is a SQL file
    sql = read_sql_file(sql_file)

    # Split the SQL into individual commands
    sql_commands = sql.split(";")

    # Format each command and return the list of commands
    sql_commands = [reformat_sql(command) for command in sql_commands]

    return sql_commands


def _add_quotes_to_list(cl: list) -> list:
    """
    Add quotes to each item in the list.

    Args:
        cl (list): List of items.

    Returns:
        list: List of items with quotes.
    """
    return [f"'{item}'" for item in cl]


def convert_list_to_sql_str(cl: list) -> str:
    """
    Convert list to SQL string.

    Args:
        cl (list): List of items.

    Returns:
        str: String of items with quotes and commas.
    """
    return ", ".join(_add_quotes_to_list(cl))
