import nltk
from nltk.corpus import brown
import copy
import viterbi

START = '*'

END = "."

PREV = "prev"

PROB = "prob"


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


def get_word_full_list(corpus: list) -> set:
    word_tag_list = get_word_tag_full_list(corpus)
    return set([word_tag[0] for word_tag in word_tag_list])


def get_tag_full_list(corpus: list) -> set:
    word_tag_list = get_word_tag_full_list(corpus)
    return set([word_tag[1] for word_tag in word_tag_list])


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


def calculate_emission(corpus: list) -> dict:
    probability_map = dict()
    for word, tag in corpus:
        if tag in probability_map:
            if word in probability_map[tag]:
                probability_map[tag][word] += 1
            else:
                probability_map[tag][word] = 1
        else:
            probability_map[tag] = dict()
            probability_map[tag][word] = 1
    for tag, tag_dict in probability_map.items():
        tag_counts = sum(tag_dict.values())
        for word in tag_dict:
            probability_map[tag][word] /= tag_counts
    return probability_map


def calculate_transmission(corpus: list) -> dict():
    probability_map = dict()
    first_word, first_tag = corpus[0]
    for second_word, second_tag in corpus[1:]:
        if first_tag in probability_map:
            if second_tag in probability_map[first_tag]:
                probability_map[first_tag][second_tag] += 1
            else:
                probability_map[first_tag][second_tag] = 1
        else:
            probability_map[first_tag] = dict()
            probability_map[first_tag][second_tag] = 1
        first_tag = second_tag
    for first_tag, first_tag_dict in probability_map.items():
        first_tag_counts = sum(first_tag_dict.values())
        for second_tag in first_tag_dict:
            probability_map[first_tag][second_tag] /= first_tag_counts
    return probability_map


def calculate_emission_with_laplace(corpus: list) -> dict:
    probability_map = dict()
    words_set = set()
    for word, tag in corpus:
        words_set.add(word)
    for word, tag in corpus:
        if tag not in probability_map:
            probability_map[tag] = dict()
            for word in words_set:
                probability_map[tag][word] = 0
    for word, tag in corpus:
        probability_map[tag][word] += 1
    for tag, tag_dict in probability_map.items():
        tag_counts = sum(tag_dict.values())
        for word in tag_dict:
            probability_map[tag][word] = (probability_map[tag][word] + 1) / (tag_counts + len(words_set))
    return probability_map

def viterbi_algorithm(sentence: list,
                      transition_matrix: dict,
                      emission_matrix: dict) -> list:
    """
    :param sentence: the corpus
    :param transition_matrix: for
    :param emission_matrix:
    :return:
    """
    # init
    sentence = ['START'] + sentence
    pi = {(0, START): 1}
    bp = {}
    S = [set([]) for i in range(len(sentence))]
    S[0].add(START)
    states = set([tag_word[0] for tag_word in transition_matrix])
    for k in range(1, len(sentence)):
        S[k] = states
    for k in range(1, len(sentence)):
        for v in S[k]:
            max_pi = 0
            max_u = 0
            for u in S[k - 1]:
                if sentence[k] not in emission_matrix[v]:  ## Todo: if a word wasn't in the corpus then e(x|y) = 0
                    emission_matrix[v][sentence[k]] = 0
                if pi[(k - 1, u)] * transition_matrix[u][v] * emission_matrix[v][sentence[k]] > max_pi:
                    max_pi = pi[(k - 1, u)] * transition_matrix[u][v] * emission_matrix[v][sentence[k]]
                    max_u = u
            bp[(k, v)] = max_u
            pi[(k, v)] = max_pi
    max_set = 0
    max_v = 0
    for i in range(1, len(sentence)):
        v = S[i]
        if pi[(len(sentence), v)] * transition_matrix[v][END] > max_set:
            max_set = pi[(len(sentence), v)] * transition_matrix[v][END]
            max_v = v
    tags_list = [0] * (len(sentence) + 1)
    tags_list[len(sentence)] = max_v
    for k in range(len(sentence), 1):
        tags_list[k] = bp[k + 1, tags_list[k + 1]]
    return tags_list


# known_words_error, unknown_words_error, total_error = most_likely_tag(get_word_tag_full_list(get_train_set()), get_word_tag_full_list(get_test_set()))
# print(known_words_error, unknown_words_error, total_error)

emission = calculate_emission(get_word_tag_full_list(get_train_set()))
transmission = calculate_transmission(get_word_tag_full_list(get_train_set()))
# emission_with_laplace = calculate_emission_with_laplace(get_word_tag_full_list(get_train_set()))
viterbi_algorithm(["The", "dog", "ate", "my", "homework", END], transmission, emission)