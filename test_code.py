import os


def simple_function(x, y):
    """
    This is a simple function.
    It adds two numbers.
    """
    result = x + y
    print(result)
    return result


class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.validate()

    def validate(self):
        if not self.data:
            raise ValueError("Data is empty")

    def complex_processing(self, multiplier: int, use_cache: bool = True) -> list:
        # Здесь нет докстринга, парсер должен сохранить только сигнатуру
        cleaned = self._clean_data(self.data)
        if use_cache:
            cache.load_or_save(cleaned)

        results = [calculate_metric(item, multiplier) for item in cleaned]
        return results
