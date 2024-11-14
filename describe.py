import pandas as pd
from data_analysis.dataStatistics import DataStatistics
import sys


def describe(data_set: str):
    try:
        df = pd.read_csv(data_set, dtype={'Hogwarts House': str, 'First Name': str, 'Last Name': str, 'Birthday': str, 'Best Hand': str})
    except FileNotFoundError:
        print(f"Error: The file '{data_set}' was not found.")
        return
    # print(df.describe().to_string())
    stats = DataStatistics(df)
    stats.describe()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python describe.py <data_set>")
    else:
        describe(sys.argv[1])
