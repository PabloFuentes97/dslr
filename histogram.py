import sys
from data_visualization import data_visualitation 
import pandas as pd

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python histogram.py <data_set> <course>")
    else:
        dataset = sys.argv[1]
        course = sys.argv[2]
        try:
            data = pd.read_csv(dataset, usecols=[course, 'Hogwarts House'])
        except FileNotFoundError:
            print(f"Error: The file '{dataset}' was not found.")
            sys.exit(1)
        except ValueError:
            print(f"Error: The course '{course}' was not found in the dataset.")
            sys.exit(1)
        data_visualitation.histogram(data, course)

    
    