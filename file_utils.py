import csv
import pickle
from typing import Iterable, Mapping, Any
from pathlib import Path


ROOT_DIR = Path(__file__).absolute().parent.parent


def _validate_csv_filename(filename:str) -> bool:
    return filename.endswith('.csv')


def _validate_pickle_filename(filename:str) -> bool:
    return filename.endswith('.pickle')


def write_csv_file(filename: str, field_headers: list[str], data: Iterable[Mapping[str, Any]]):
    if not _validate_csv_filename(filename):
        raise Exception(f"Filename {filename} is invalid csv")

    print(f'Writing csv file {filename}')
    with open(filename, 'w', newline='', encoding='utf-8') as csv_file:
        csv_file_writer = csv.DictWriter(csv_file, field_headers)
        _ = csv_file_writer.writeheader()
        csv_file_writer.writerows(data)
    print(f'Finish writing csv file {filename}')


def write_pickle_file(filename: str, data: dict):
    if not _validate_pickle_filename(filename):
        raise Exception(f"Filename {filename} is invalid pickle")

    print(f'Writing pickle file {filename}')
    with open(filename, 'wb') as pickle_file:
        pickle.dump(data, pickle_file)
    print(f'Finish writing pickle file {filename}')


def read_pickle_file(filename: str) -> dict:
    if not _validate_pickle_filename(filename):
        raise Exception(f"Filename {filename} is invalid pickle")

    print(f'Reading pickle file {filename}')
    with open(filename, 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    return data
