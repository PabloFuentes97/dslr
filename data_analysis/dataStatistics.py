import math


class DataStatistics:
    def __init__(self, df):
        self.df = df

    def count(self, items):
        return float(len(self.filter_items(items)))

    def mean(self, items):
        filtered_items = self.filter_items(items)
        total = sum(filtered_items)
        return total / self.count(items)

    def min(self, items):
        filtered_items = self.filter_items(items)
        min_value = filtered_items[0]
        for item in filtered_items:
            if item < min_value:
                min_value = item
        return float(min_value)

    def max(self, items):
        filtered_items = self.filter_items(items)
        max_value = filtered_items[0]
        for item in filtered_items:
            if item > max_value:
                max_value = item
        return float(max_value)

    def percentile(self, items, percentile):
        filtered_items = self.filter_items(items)
        filtered_items.sort()
        count = self.count(items)
        if count == 0:
            return None
        position = percentile * (count - 1)
        lower_index = int(position)
        upper_index = lower_index + 1
        if position == lower_index:
            return filtered_items[lower_index]
        lower_value = filtered_items[lower_index]
        upper_value = filtered_items[min(upper_index, count - 1)]
        fractional_part = position - lower_index
        return lower_value + fractional_part * (upper_value - lower_value)

    def standard_deviation(self, items):
        filtered_items = self.filter_items(items)
        mean_value = self.mean(filtered_items)
        squared_diff_sum = sum((item - mean_value) ** 2 for item in filtered_items)
        variance = squared_diff_sum / (self.count(filtered_items) - 1)
        return math.sqrt(variance)


    def get_numerical_columns(self):
        columns = self.df.select_dtypes(include=['number']).columns.tolist()
        return columns

    def get_statistics(self):
        header = self.get_numerical_columns()
        count_values = [self.count(self.df[col]) for col in header]
        mean_values = [self.mean(self.df[col]) for col in header]
        std_values = [self.standard_deviation(self.df[col]) for col in header]
        min_values = [self.min(self.df[col]) for col in header]
        percentile_25 = [self.percentile(self.df[col], 0.25) for col in header]
        percentile_50 = [self.percentile(self.df[col], 0.5) for col in header]
        percentile_75 = [self.percentile(self.df[col], 0.75) for col in header]
        max_values = [self.max(self.df[col]) for col in header]
        return header, count_values, mean_values, std_values, min_values, percentile_25, percentile_50, percentile_75, max_values

    def describe(self):
        headers, count_values, mean_values, std_values, min_values, percentile_25, percentile_50, percentile_75, max_values = self.get_statistics()
        headers.insert(0, "")
        count_values.insert(0, "Count")
        mean_values.insert(0, "Mean")
        std_values.insert(0, "std")
        min_values.insert(0, "Min")
        percentile_25.insert(0, "25%")
        percentile_50.insert(0, "50%")
        percentile_75.insert(0, "75%")
        max_values.insert(0, "Max")

        rows = [headers, count_values, mean_values, std_values, min_values, percentile_25, percentile_50, percentile_75, max_values]
        self.print_table(rows         )

    def print_table(self, rows):
        column_widths = self.calculate_column_widths(rows)
        for row in rows:
            self.print_row(row, column_widths)

    @staticmethod
    def calculate_column_widths(rows, min_width=14):
        return [
            max(len(f"{row[i]:.6f}") if isinstance(row[i], float) else len(str(row[i])) for row in rows)
            if len(rows) > 0 else min_width
            for i in range(len(rows[0]))
        ]
    @staticmethod
    def print_row(row, col_widths):
        formatted_row = "  ".join(
            f"{item:>{col_widths[i]}.6f}" if isinstance(item, float) else f"{str(item):>{col_widths[i]}}"
            for i, item in enumerate(row)
        )
        print(formatted_row)

    @staticmethod
    def filter_items(items):
        return [item for item in items if not math.isnan(item)]




