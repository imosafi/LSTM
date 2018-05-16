# -*- coding: utf-8 -*-

import unicodedata
import re


class TextCleaner:

    # 'ף' 'פ' and 'כ' 'ך'
    # requires 2 representations, the rest don't
    final_letters_dict = {'ם': 'מ', 'ן': 'נ', 'ץ': 'צ'}

    punctuation_replacement_dict = {';': ',', '–': ' ', '-': ' ', '_': ' ', '‐': ' ', '*': '', '[': '(', '{': '(',
                                    ']': ')', '}': ')', '!': '.', '\'': '', '־': ' ', '/': '', '=': '', '\u200b': '',
                                    '\xad': '', '`': '', '~': '', '“': '"', '”': '"'}

    foreign_letters_dict = {'μ': '', 'ε': '', 'ι': '', 'ω': '', 'α': '', 'β': '', 'λ': '', 'ψ': '', 'Α': '', 'Ђ': '',
                            'Е': '', 'С': '', 'а': '', 'б': '', 'в': '', 'д': '', 'е': '', 'и': '', 'л': '', 'м': '',
                            'н': '', 'о': '', 'п': '', 'р': '', 'с': '', 'т': '', 'х': '', 'ч': '', 'ъ': '', 'ы': '',
                            'я': '', 'ѣ': '', 'к': '', 'ь': '', 'У': '', 'г': '', 'ц': '', 'Д': '', 'у': '', '\\': '',
                            '<': ''}

    # def __init__(self):

    def remove_diacritic(self, text):
        return ''.join([c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c)])

    def replace_final_letters(self, text):
        return ''.join([self.final_letters_dict[c] if c in self.final_letters_dict.keys() else c for c in text])

    def remove_foreign_letters(self, text):
        return ''.join([self.foreign_letters_dict[c] if c in self.foreign_letters_dict.keys() else c for c in text])

    def combine_multiple_spaces(self, text):
        return ' '.join(text.split())

    def remove_reference_numbers(self, text):
        return re.sub('\[[0-9]*\]', '', text)

    def replace_useless_punctuation(self, text):
        return ''.join(
            [self.punctuation_replacement_dict[c] if c in self.punctuation_replacement_dict.keys() else c for c in text])

    def remove_all_numbers(self, text):
        return ''.join(['' if c.isdigit() else c for c in text])

    def remove_english_letters(self, text):
        return re.sub('[a-zA-Z]', '', text)

    def combine_dots(self, text):
        return re.sub('[.]+', '.', text)

    def remove_empty_parentheses(self, text):
        return re.sub('\([\s,.]*\)', '', text)

    def clean(self, text):
        cleaned_text = self.remove_diacritic(text)
        cleaned_text = self.remove_reference_numbers(cleaned_text)
        cleaned_text = self.remove_all_numbers(cleaned_text)
        cleaned_text = self.replace_useless_punctuation(cleaned_text)
        cleaned_text = self.remove_foreign_letters(cleaned_text)
        cleaned_text = self.remove_english_letters(cleaned_text)
        cleaned_text = self.combine_multiple_spaces(cleaned_text)
        cleaned_text = self.remove_empty_parentheses(cleaned_text)
        cleaned_text = self.combine_dots(cleaned_text)
        cleaned_text = self.replace_final_letters(cleaned_text)
        return cleaned_text