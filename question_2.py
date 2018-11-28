import nltk
from nltk.corpus import brown
import copy
import viterbi
import random

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


def get_train_set() -> list:  # a
    train_set = news_tagged._pieces[:(int(len(news_tagged._pieces) * 0.9))]
    return train_set


def get_test_set() -> list:  # a
    test_set = news_tagged._pieces[int(len(news_tagged._pieces) * 0.9):]
    return test_set


def train_most_likely_tag(corpus: list) -> dict:  # b.1
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


def most_likely_tag(train_set: list, test_set: list) -> (float, float, float):  # b.2
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


def calculate_emission(corpus: list) -> dict:  # c.1
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


def calculate_transmission(corpus: list) -> dict():  # c.1
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


def calculate_emission_with_laplace(train_set: list, test_set: list) -> dict:  # d.1
    set_of_distinct_words = set()
    for word, tag in train_set + test_set:
        set_of_distinct_words.add(word)
    num_of_distinct_words = len(set_of_distinct_words)
    probability_map = dict()
    words_set = set()
    for word, tag in train_set:
        words_set.add(word)
    for word, tag in train_set:
        if tag not in probability_map:
            probability_map[tag] = dict()
            for word in words_set:
                probability_map[tag][word] = 0
    for word, tag in train_set:
        probability_map[tag][word] += 1
    for tag, tag_dict in probability_map.items():
        tag_counts = sum(tag_dict.values())
        for word in tag_dict:
            probability_map[tag][word] = (probability_map[tag][word] + 1) / (tag_counts + num_of_distinct_words)
    return probability_map


def viterbi_algorithm(sentence: list,
                      transition_matrix: dict,
                      emission_matrix: dict) -> list:  # c.2
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
    states = set([tag_word for tag_word in transition_matrix])
    for k in range(1, len(sentence)):
        S[k] = states
    did_all_the_word_appeared, last_good_word_index = viterbi_recursion(S, bp, emission_matrix, pi, sentence,
                                                                        transition_matrix)
    max_set, max_v = decide_last_word_tag(S, last_good_word_index, pi, transition_matrix)
    tags_list = [''] * last_good_word_index
    tags_list[last_good_word_index - 1] = max_v
    for k in range(last_good_word_index - 2, -1, -1):
        print('k ' +str(k))
        tags_list[k] = (bp[k + 1, tags_list[k + 1]])
    if did_all_the_word_appeared:
        return tags_list
    else:
        all_tags = S[1]
        return tags_list + random.sample(all_tags, len(sentence) - last_good_word_index)


def decide_last_word_tag(S, last_good_word_index, pi, transition_matrix):
    max_set = 0
    all_tags = S[1]
    max_v = random.sample(all_tags, 1)[0]  # define the tag as random by default - would be replace is there is better
    sentence_len = last_good_word_index
    for v in all_tags:
        if END not in transition_matrix[v] or (sentence_len - 1, v) not in pi:
            continue
        if pi[(sentence_len - 1, v)] * transition_matrix[v][END] > max_set:
            max_set = pi[(sentence_len - 1, v)] * transition_matrix[v][END]
            max_v = v

    return max_set, max_v



def viterbi_recursion(S, bp, emission_matrix, pi, sentence, transition_matrix):
    """
    
    :param S: 
    :param bp: 
    :param emission_matrix: 
    :param pi: 
    :param sentence: 
    :param transition_matrix: 
    :return: If all the word appreard, the last word that was good (len of sentence if all them)
    """
    for k in range(1, len(sentence)):
        did_word_k_appeared = False
        for v in S[k]:
            max_pi = 0
            max_u = 0
            for u in S[k - 1]:
                if sentence[k] not in emission_matrix[v]:  # Todo: if a word wasn't in the corpus then e(x|y) = 0
                    emission_matrix[v][sentence[k]] = 0
                if v not in transition_matrix[u]:
                    transition_matrix[u][v] = 0

                if pi[(k - 1, u)] * transition_matrix[u][v] * emission_matrix[v][sentence[k]] > max_pi:
                    did_word_k_appeared = True
                    max_pi = pi[(k - 1, u)] * transition_matrix[u][v] * emission_matrix[v][sentence[k]]
                    max_u = u
                    print("max pi: " + str(max_pi))
                    print("max u for v: " + u + " " + v)
            if max_pi == 0:
                bp[(k, v)] = random.sample(S[k], 1)[0]
            else:
                bp[(k, v)] = max_u
            pi[(k, v)] = max_pi
        if not did_word_k_appeared:
            return did_word_k_appeared, k

    return True, len(sentence)


def calculate_with_viterbi():  # c.3
    test_set = get_word_tag_full_list(get_test_set())
    emission = calculate_emission(get_word_tag_full_list(get_train_set()))
    transmission = calculate_transmission(get_word_tag_full_list(get_train_set()))
    words_in_line = list()
    correct_tags = list()
    for word, tag in test_set:
        words_in_line.append(word)
        correct_tags.append(tag)
    predicted_tags = viterbi_algorithm(words_in_line, transmission, emission)
    num_of_mistakes = 0
    for i in range(len(predicted_tags)):
        if predicted_tags[i] != correct_tags[i]:
            num_of_mistakes += 1
    error_rate = num_of_mistakes / len(predicted_tags)
    return error_rate


def calculate_with_viterbi_laplace():  # d.2
    test_set = get_word_tag_full_list(get_test_set())
    emission = calculate_emission_with_laplace(get_word_tag_full_list(get_train_set()))
    transmission = calculate_transmission(get_word_tag_full_list(get_train_set()))
    words_in_line = list()
    correct_tags = list()
    for word, tag in test_set:
        words_in_line.append(word)
        correct_tags.append(tag)
    predicted_tags = viterbi_algorithm(words_in_line, transmission, emission)
    num_of_mistakes = 0
    for i in range(len(predicted_tags)):
        if predicted_tags[i] != correct_tags[i]:
            num_of_mistakes += 1
    error_rate = num_of_mistakes / len(predicted_tags)
    return error_rate


def get_set_of_pseudo_words(train_set: list, test_set: list):  # e.1
    words_counts = dict()
    for word, tag in train_set:
        if word in words_counts:
            words_counts += 1
        else:
            words_counts[word] = 1
    unknown_words = list()
    for word, tag in test_set:
        if word not in words_counts:
            unknown_words.append(word)
    low_frequency_words = list()
    for key in train_set:
        if words_counts[key] < 3:
            low_frequency_words.append(key)
    return unknown_words, low_frequency_words





# known_words_error, unknown_words_error, total_error = most_likely_tag(get_word_tag_full_list(get_train_set()), get_word_tag_full_list(get_test_set()))
# print(known_words_error, unknown_words_error, total_error)

emission = calculate_emission(get_word_tag_full_list(get_train_set()))
transmission = calculate_transmission(get_word_tag_full_list(get_train_set()))
# emission_with_laplace = calculate_emission_with_laplace(get_word_tag_full_list(get_train_set()), get_word_tag_full_list(get_test_set()))
a = viterbi_algorithm(["The", "jury", "uykgiuguyg", "it", "did"], transmission, emission)
print(a)