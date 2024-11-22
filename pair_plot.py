import sys
from data_visualization import data_visualitation 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python histogram.py <data_set>")
    else:
        dataset = sys.argv[1]
        try:
            data = pd.read_csv(dataset)
        except FileNotFoundError:
            print(f"Error: The file '{dataset}' was not found.")
            sys.exit(1)
        except ValueError:
            print(f"Error: one of the courses was not found in the dataset.")
            sys.exit(1)
        sns.pairplot(data, hue="Hogwarts House")
        plt.show()