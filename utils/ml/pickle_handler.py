# Write to a pickle file
import os
import pickle
import logging


logger = logging.getLogger(__name__)


# Write to a pickle file
def write_to_pickle(data: object, pickle_file: str):
    """
    Write data to a pickle file
    :param data:
    :param pickle_file:
    :return: None
    """
    try:
        # Check if the file is a pickle file
        if not pickle_file.endswith('.pkl'):
            logger.error(f"Invalid file format. Expected a pickle file. {pickle_file}")
            raise ValueError("Invalid file format. Expected a pickle file.")

        # Check if the directory exists
        if not os.path.exists(os.path.dirname(pickle_file)):
            logger.debug(f"Creating directory: {os.path.dirname(pickle_file)}")
            os.makedirs(os.path.dirname(pickle_file))

        # Write data to a pickle file
        logger.debug(f"Writing data to pickle file: {pickle_file}")
        with open(pickle_file, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        logger.error(f"Error in writing to a pickle file {pickle_file}:{e}")
        raise e
