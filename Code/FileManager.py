import json
from enum import Enum
from pathlib import Path
import os
from TextCleaner import TextCleaner


class TextType(Enum):
    TRAIN = 1
    TEST = 2

class FileManager:

    def __init__(self):
        self.text_cleaner = TextCleaner()
        self.configuration_path = 'data/config.json'
        self.configuration_data = self.load_configuration_file()
        self.training_data_path = self.configuration_data['training_data_path']
        self.testing_data_path = self.configuration_data['testing_data_path']
        self.cleaned_training_data_path = self.configuration_data['cleaned_training_data_path']
        self.cleaned_testing_data_path = self.configuration_data['cleaned_testing_data_path']
        self.should_load_trained_model = self.configuration_data['load_trained_model']
        self.trained_model_path = self.configuration_data['trained_model_path']


    def load_configuration_file(self):
        with open(self.configuration_path, 'r') as f:
            return json.load(f)

    def save_data(self, data, file_path):
        f = open(file_path, 'w',encoding='utf8')
        if (type(data) is list):
            for item in data:
                f.write("%s\n" % item)
        else:
            f.write(data)
        f.close()

    def get_cleaned_text(self, text_type):
        if text_type == TextType.TRAIN:
            if Path(self.cleaned_training_data_path).is_file():
                with open(self.cleaned_training_data_path, 'r', encoding='utf8') as f:
                    return f.read()
            else:
                with open(self.training_data_path, 'r', encoding='utf8') as f:
                    cleaned_data = self.text_cleaner.clean(f.read())
                    self.save_data(cleaned_data, self.cleaned_training_data_path)
                    return cleaned_data
        else:
            if Path(self.cleaned_testing_data_path).is_file():
                with open(self.cleaned_testing_data_path, 'r', encoding='utf8') as f:
                    return f.read()
            else:
                with open(self.testing_data_path, 'r', encoding='utf8') as f:
                    cleaned_data = self.text_cleaner.clean(f.read())
                    self.save_data(cleaned_data, self.cleaned_testing_data_path)
                    return cleaned_data

    def delete_processed_data(self):
        os.remove(self.cleaned_training_data_path)
        os.remove(self.cleaned_testing_data_path)

    def save_results(self, accuracy, cross_entropy, generated_text):
        f = open('data/output.txt', 'w', encoding='utf8')
        f.write("accuracy: %s\n" % accuracy)
        f.write("cross entropy: %s\n" % cross_entropy)
        f.write("generated text:\n %s" % generated_text)
        f.close()