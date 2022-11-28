import pandas as pd

FILE_DIRECTORY = "../data"


def load_forbes_df(year: int, columns: list, sep: str = ','):
    df = pd.read_csv(
        f'{FILE_DIRECTORY}/Forbes Global 2000 - {year}.csv',
        sep=sep,
        error_bad_lines=False
    )
    df.columns = columns
    df['year'] = year
    return df