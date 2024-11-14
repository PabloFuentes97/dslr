import pandas as pd
import matplotlib.pyplot as plt

def histogram(data: pd.DataFrame, course: str):
    houses = data['Hogwarts House'].unique()
    plt.figure(figsize=(10, 6))
    for house in houses:
        subset = data[data['Hogwarts House'] == house]
        plt.hist(subset[course].dropna(), bins=20, alpha=0.5, label=house)
    
    plt.title(f'Histogram of {course} Scores by Hogwarts House')
    plt.xlabel(f'{course} Score')
    plt.ylabel('Frequency')
    plt.legend(title='Hogwarts House')
    plt.show()

def scatter_plot(data: pd.DataFrame, course1: str, course2: str):
    houses = data['Hogwarts House'].unique()
    plt.figure(figsize=(10, 6))
    for house in houses:
        subset = data[data['Hogwarts House'] == house]
        plt.scatter(subset[course1], subset[course2], alpha=0.5, label=house)
    
    plt.title(f'Scatter Plot of {course1} vs {course2} by Hogwarts House')
    plt.xlabel(f'{course1} Score')
    plt.ylabel(f'{course2} Score')
    plt.legend(title='Hogwarts House')
    plt.show()
