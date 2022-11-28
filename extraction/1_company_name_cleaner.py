import csv
import logging
import sys

from tqdm import tqdm
from pathlib import Path

from file_utils import write_pickle_file
from src.extraction.utils import get_text_similarity


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    '%m-%d-%Y %H:%M:%S'
)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

ROOT_DIR = Path(__file__).absolute().parent.parent
DATA_PATH = f'{ROOT_DIR}/data/raw'

YEARS = list(range(2008, 2021))


def get_file_name(year: int) -> str:
    return f'{DATA_PATH}/Forbes Global 2000 - {year}.csv'


class Similarity:
    def __init__(self, word: str, other: str, score: float) -> None:
        self.word = word
        self.other = other
        self.score = score

    def __str__(self) -> str:
        return f'({self.word}, {self.other})={self.score}'

    def to_dict(self) -> dict:
        return {
            'word': self.word,
            'other': self.other,
            'score': self.score
        }


similarities = {}

for year in tqdm(YEARS):
    filename = get_file_name(year)
    logger.info(f'Processing file {filename}')

    with open(filename, 'r') as ranking_file:
        reader = csv.DictReader(ranking_file)

        for row in reader:
            new_company = str(row['Company'])
            new_company = new_company.lower()

            # Check if company is not already on our map
            if not similarities.get(new_company):

                # If company is not on similarities
                # we want to check all current similarities
                for key in similarities.keys():
                    if key != new_company:
                        score = get_text_similarity(key, new_company)
                        logger.debug(f'Similarity score for {key}-{new_company}={score}')

                        similarity = Similarity(
                            word=key,
                            other=new_company,
                            score=score
                        )

                        key_stored_similarity = similarities[key]

                        # We replace previous similarity
                        # if new similarity has a better score
                        # or if stored similarity is None
                        if not key_stored_similarity or score > key_stored_similarity.score:
                            logger.debug(f'Updating {similarity} on map')
                            similarities[key] = similarity

                # We store the company name with an empty Similarity
                logger.debug(f'Storing empty value for {new_company}')
                similarities[new_company] = None

        logger.info(f'Finished processing {filename}')

for k, v in similarities.items():
    print(v)
    similarities[k] = {} if not v else v.to_dict()

write_pickle_file(
    filename='company_similarities.pickle',
    data=similarities
)


