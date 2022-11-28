from file_utils import read_pickle_file

FILENAME = 'company_similarities.pickle'

similarities = read_pickle_file(FILENAME)

for k,v in similarities.items():
    if v['score'] > 0.5:
        print(v)
