import pandas as pd
from data_analysis.dataStatistics import DataStatistics

data = pd.read_csv('dataset_train.csv')

courses = ['Arithmancy', 'Astronomy', 'Herbology', 'Defense Against the Dark Arts',
           'Divination', 'Muggle Studies', 'Ancient Runes', 'History of Magic',
           'Transfiguration', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying']


std_dev_results = {}


for course in courses:
    min_score = DataStatistics(data).min(data[course])
    max_score = DataStatistics(data).max(data[course])
    scaled_scores = 10 * (data[course] - min_score) / (max_score - min_score)
    
    std_dev = DataStatistics(data).standard_deviation(scaled_scores.dropna())
    std_dev_results[course] = std_dev

sorted_std_dev_results = dict(sorted(std_dev_results.items(), key=lambda item: item[1], reverse=True))

for course, std_dev in sorted_std_dev_results.items():
    print(f" {course}: {std_dev:.2f}")
