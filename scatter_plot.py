import sys
from data_visualization import data_visualitation 
import pandas as pd



if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python histogram.py <data_set> <course1> <course2>")
    else:
        dataset = sys.argv[1]
        course_1 = sys.argv[2]
        course_2 = sys.argv[3]
        try:
            data = pd.read_csv(dataset, usecols=[course_1, course_2, 'Hogwarts House'])
        except FileNotFoundError:
            print(f"Error: The file '{dataset}' was not found.")
            sys.exit(1)
        except ValueError:
            print(f"Error: one of the courses was not found in the dataset.")
            sys.exit(1)
        data_visualitation.scatter_plot(data, course_1, course_2)