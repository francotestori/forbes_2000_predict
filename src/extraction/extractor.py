import argparse
from typing import List

import numpy as np
import pandas as pd
import unidecode as unidecode
from pandas._libs.missing import NAType

from extraction.continents import CONTINENTS
from extraction.rebrand import REBRAND


class ForbesExtractor:
    def __init__(self, year: int, sep: str = ',') -> None:
        super().__init__()
        self.year = year
        self.sep = sep
        self._set_columns()

    def run(self, data_source_path: str) -> pd.DataFrame:
        file_name: str = f'{data_source_path}/Forbes Global 2000 - {self.year}.csv'

        df: pd.DataFrame = pd.read_csv(
            file_name,
            sep=self.sep,
            on_bad_lines='warn'
        )

        df.columns = self.columns
        df['year'] = self.year

        df = self._curate_money_values(df)
        df = self._curate_percentage_values(df)
        df = self._set_column_types(df)

        return df

    def _set_columns(self):
        if self.year == 2020:
            self.columns = ['rank', 'company', 'country', 'sales', 'profits', 'assets', 'market_value', 'sector',
                            'industry']
        elif self.year >= 2015:
            self.columns = ['company', 'market_value', 'sales', 'profits', 'assets', 'rank', 'sector', 'industry',
                            'continent', 'country', 'headquarters', 'state', 'ceo', 'forbes_webpage',
                            'profits_%_assets', 'profits_%_sales']
        elif self.year == 2014:
            self.columns = ['company', 'sector', 'industry', 'continent', 'country', 'market_value', 'sales', 'profits',
                            'assets', 'rank', 'forbes_webpage', 'profits_%_assets', 'profits_%_sales']
        elif self.year >= 2011:
            self.columns = ['company', 'industry', 'country', 'market_value', 'profits', 'assets', 'sales', 'rank',
                            'forbes_webpage', 'profits_%_assets', 'profits_%_sales']
        elif self.year >= 2009:
            self.columns = ['company', 'industry', 'country', 'market_value', 'profits', 'assets', 'sales', 'rank',
                            'profits_%_assets', 'profits_%_sales']
        else:
            self.columns = ['company', 'industry', 'country', 'market_value', 'profits', 'assets', 'sales', 'rank']

    @staticmethod
    def _set_column_types(df: pd.DataFrame) -> pd.DataFrame:
        def _set_col_type(dataframe: pd.DataFrame, column: str, col_type: str) -> None:
            if column in df.columns:
                dataframe[column] = dataframe[column].astype(col_type)

        # Set string column types
        _set_col_type(dataframe=df, column='company', col_type='string')
        _set_col_type(dataframe=df, column='industry', col_type='string')
        _set_col_type(dataframe=df, column='country', col_type='string')
        _set_col_type(dataframe=df, column='forbes_webpage', col_type='string')
        _set_col_type(dataframe=df, column='sector', col_type='string')
        _set_col_type(dataframe=df, column='continent', col_type='string')
        _set_col_type(dataframe=df, column='headquarters', col_type='string')
        _set_col_type(dataframe=df, column='state', col_type='string')
        _set_col_type(dataframe=df, column='ceo', col_type='string')

        # Set numeric float column types
        _set_col_type(dataframe=df, column='market_value', col_type='float')
        _set_col_type(dataframe=df, column='profits', col_type='float')
        _set_col_type(dataframe=df, column='assets', col_type='float')
        _set_col_type(dataframe=df, column='sales', col_type='float')
        _set_col_type(dataframe=df, column='profits_%_assets', col_type='float')
        _set_col_type(dataframe=df, column='profits_%_sales', col_type='float')

        # Set numeric integer (base64) column types
        _set_col_type(dataframe=df, column='rank', col_type='Int64')
        _set_col_type(dataframe=df, column='year', col_type='Int64')

        return df

    @staticmethod
    def _curate_percentage_values(df: pd.DataFrame) -> pd.DataFrame:
        def _clean_infinite_values(percentage):
            return np.nan if percentage in ['∞', '-∞'] else percentage

        percentage_columns = ['profits_%_assets', 'profits_%_sales']
        for column in percentage_columns:
            if column in df.columns:
                df[column] = df[column].apply(_clean_infinite_values)

        return df

    @staticmethod
    def _curate_money_values(df: pd.DataFrame) -> pd.DataFrame:
        def money_value_to_float(value) -> float:
            money_value: str = str(value)\
                .replace('$', '')\
                .replace(' ', '')

            if 'B' in money_value:
                money_value = money_value \
                    .replace('B', '') \
                    .replace(',', '')
            elif 'M' in money_value:
                aux = money_value.replace(' ', '') \
                    .replace('M', '') \
                    .replace('.', '')

                # Value is negative
                if '-' in aux:
                    aux = aux.replace('-', '')
                    money_value = f'-0.{aux}'
                else:
                    money_value = f'0.{aux}'

            return float(money_value)

        money_columns: List[str] = ['market_value', 'profits', 'assets', 'sales']
        for column in money_columns:
            df[column] = df[column].map(money_value_to_float)

        return df


def unify_company_names(df: pd.DataFrame) -> pd.DataFrame:
    df['company'] = df['company'].replace(REBRAND, inplace=False)

    for company in sorted(df['company'].unique()):
        company_name = str(company)
        company_name_lower = company_name.lower().strip()
        lower_case = unidecode.unidecode(company_name_lower)
        df.loc[df['company'] == company, 'company'] = lower_case

    return df


def fill_empty_values_from_company_tuples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill column values from previous record when based on the selected company column value.
    """
    def prepare_values_and_fill_if_missing(dataframe: pd.DataFrame, col: str, col_type: str):
        reference = dataframe.groupby(['company', 'country'], sort=False)[col].first().reset_index()
        dictionary = {}
        for index, row in reference.iterrows():
            if not isinstance(row[col], NAType):
                dictionary[row['company']] = {
                    row['country']: row[col]
                }

        dataframe[col] = dataframe[['company', 'country']].apply(
            lambda x: dictionary.get(x['company'], {}).get(x['country'], pd.NA), axis=1
        )
        dataframe[col] = dataframe[col].astype(col_type)

        return dataframe

    df = prepare_values_and_fill_if_missing(
        dataframe=df,
        col='sector',
        col_type='string'
    )

    df = prepare_values_and_fill_if_missing(
        dataframe=df,
        col='industry',
        col_type='string'
    )

    df = prepare_values_and_fill_if_missing(
        dataframe=df,
        col='ceo',
        col_type='string'
    )

    df = prepare_values_and_fill_if_missing(
        dataframe=df,
        col='forbes_webpage',
        col_type='string'
    )

    df = prepare_values_and_fill_if_missing(
        dataframe=df,
        col='state',
        col_type='string'
    )

    df = prepare_values_and_fill_if_missing(
        dataframe=df,
        col='headquarters',
        col_type='string'
    )

    # Fill continent based on company
    continent_map = {}
    for index, row in df.groupby('country', sort=False)['continent'].first().reset_index().iterrows():
        if not isinstance(row['continent'], NAType):
            continent_map[row['country']] = row['continent']

    df['continent'] = df.apply(lambda x: continent_map.get(x['country'], pd.NA), axis=1)
    df['continent'] = df['country'].replace(CONTINENTS, inplace=False)
    df['continent'] = df['continent'].astype('string')

    return df


def main():
    parser = argparse.ArgumentParser(description="Bidder application")
    parser.add_argument(
        "--data-source-path",
        dest="data_source_path",
        type=str,
        help="Path to the data dyrectory where to find Forbes raw data files",
    )
    parser.add_argument(
        "--output-storage-path",
        dest="output_storage_path",
        type=str,
        help="Path to the data dyrectory where to find Forbes raw data files",
    )
    parser.add_argument(
        "--year-from",
        dest="year_from",
        type=int,
        help="Year from when to analyze data",
    )
    parser.add_argument(
        "--year-to",
        dest="year_to",
        type=int,
        help="Year to when to analyze data",
    )

    args = parser.parse_args()
    step: int = 1

    dataframes = []
    nrows = 0
    for year in range(args.year_from, args.year_to, step):
        extractor = ForbesExtractor(year=year)
        df: pd.DataFrame = extractor.run(args.data_source_path)

        df_nrows = len(df.index)
        nrows = nrows + df_nrows
        print(f'dataframe for year {year} has {df_nrows} rows')

        dataframes.append(df)

    print(f'Final dataframe should have {nrows} rows')

    # Merge all dataframes into a combined dataframe
    final_df: pd.DataFrame = pd.concat(dataframes, axis=0)
    print(f'Final dataframe has {len(final_df.index)} rows')

    # Unify values based on company name
    final_df = unify_company_names(final_df)

    final_df = fill_empty_values_from_company_tuples(final_df)

    output_storage_file_path: str = f'{args.output_storage_path}/forbes_merged_raw_{args.year_from}_{args.year_to}.csv'
    final_df.to_csv(output_storage_file_path, index=False, header=True)


if __name__ == "__main__":
    main()