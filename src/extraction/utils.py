import pandas as pd
from difflib import SequenceMatcher


def read_file(file_name: str, columns: list, sep: str = ',') -> pd.DataFrame:
    """
    Read a specified csv file and return a pandas Dataframe object from it's data.
    You also need to specify the desired dataframe column names.
    You might also specify a custom separator string (default is `,`)

    :param file_name:
    :param columns:
    :param sep:
    :return:
    """
    df = pd.read_csv(
        file_name,
        sep=sep,
        error_bad_lines=False
    )
    df.columns = columns
    return df


def get_text_similarity(a: str, b: str) -> float:
    """
    Finds the text similarity between two strings.

    Note that this is 1 if the sequences are identical,
    and 0 if they have nothing in common.

    We acknowledge that .ratio() is expensive to compute.

    - param a:
    - param b:
    return: a float with the corresponding similarity ratio score
    """
    return SequenceMatcher(None, a, b).ratio()
