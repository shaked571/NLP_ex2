import nltk
from nltk.corpus import brown
import copy

# nltk.download('brown')
news_tagged = brown.tagged_sents(categories='news')  # Download all the data


def get_word_tag_full_list(corpus: list) -> list:
    """
    :return: the training data as one list of tags
    """
    word_tag_list = []
    for file in corpus:
        for line in file:
            for couple in line:
                word_tag_list.append(couple)
    return copy.deepcopy(word_tag_list)


def get_train_set() -> list:
    train_set = news_tagged._pieces[:(int(len(news_tagged._pieces) * 0.9))]
    return train_set


def get_test_set() -> list:
    test_set = news_tagged._pieces[int(len(news_tagged._pieces) * 0.9):]
    return test_set


def train_most_likely_tag(corpus: list) -> dict:
    probability_map = dict()
    for word, tag in corpus:
        if word in probability_map:
            if tag in probability_map[word]:
                probability_map[word][tag] += 1
            else:
                probability_map[word][tag] = 1
        else:
            probability_map[word] = dict()
            probability_map[word][tag] = 1
    return probability_map


def calculate_most_likely_tag(word_tag_dict) -> str:
    return max(word_tag_dict, key=word_tag_dict.get)


def most_likely_tag(train_set: list, test_set: list) -> (float, float, float):
    known_words_prediction_right = 0
    known_words_prediction_wrong = 0
    unknown_words_prediction_right = 0
    unknown_words_prediction_wrong = 0
    probability_map = train_most_likely_tag(train_set)
    for word, tag in test_set:
        if word in probability_map:
            predicted_tag = calculate_most_likely_tag(probability_map[word])
            if predicted_tag == tag:
                known_words_prediction_right += 1
            else:
                known_words_prediction_wrong += 1
        else:
            predicted_tag = 'NN'
            if predicted_tag == tag:
                unknown_words_prediction_right += 1
            else:
                unknown_words_prediction_wrong += 1
    known_words_error_rate = known_words_prediction_wrong / (known_words_prediction_wrong + known_words_prediction_right)
    unknown_words_error_rate = unknown_words_prediction_wrong / (unknown_words_prediction_wrong + unknown_words_prediction_right)
    total_error_rate = (known_words_prediction_wrong + unknown_words_prediction_wrong) / len(test_set)
    return known_words_error_rate, unknown_words_error_rate, total_error_rate


known_words_error, unknown_words_error, total_error = most_likely_tag(get_word_tag_full_list(get_train_set()), get_word_tag_full_list(get_test_set()))
print(known_words_error, unknown_words_error, total_error)


