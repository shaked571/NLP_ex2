import operator

import question_2
import numpy as np

START = '*'

PREV = "prev"

PROB = "prob"

my_train_data = question_2.get_word_tag_train_full_list()
for couple in my_train_data:
    print(couple)

#
# def viterbi(transition_probabilities, conditional_probabilities):
#     # Initialise everything
#     num_samples = conditional_probabilities.shape[1]
#     num_states = transition_probabilities.shape[0]  # number of states
#
#     c = np.zeros(num_samples)  # scale factors (necessary to prevent underflow)
#     viterbi = np.zeros((num_states, num_samples))  # initialise viterbi table
#     best_path_table = np.zeros((num_states, num_samples))  # initialise the best path table
#     best_path = np.zeros(num_samples).astype(np.int32)  # this will be your output
#
#     # B- appoint initial values for viterbi and best path (bp) tables - Eq (32a-32b)
#     viterbi[:, 0] = conditional_probabilities[:, 0]
#     c[0] = 1.0 / np.sum(viterbi[:, 0])
#     viterbi[:, 0] = c[0] * viterbi[:, 0]  # apply the scaling factor
#
#     # C- Do the iterations for viterbi and psi for time>0 until T
#     for t in range(1, num_samples):  # loop through time
#         for s in range(0, num_states):  # loop through the states @(t-1)
#             trans_p = viterbi[:, t - 1] * transition_probabilities[:, s]  # transition probs of each state transitioning
#             best_path_table[s, t], viterbi[s, t] = max(enumerate(trans_p), key=operator.itemgetter(1))
#             viterbi[s, t] = viterbi[s, t] * conditional_probabilities[s][t]
#
#         c[t] = 1.0 / np.sum(viterbi[:, t])  # scaling factor
#         viterbi[:, t] = c[t] * viterbi[:, t]
#
#     ## D - Back-tracking
#     best_path[num_samples - 1] = viterbi[:, num_samples - 1].argmax()  # last state
#     for t in range(num_samples - 1, 0, -1):  # states of (last-1)th to 0th time step
#         best_path[t - 1] = best_path_table[best_path[t], t]
#     return best_path


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
    sentence = ['dummy'] + sentence
    pi = {(0, START): 1}
    bp = {}
    S = []
    S[0] = {START}
    states = set([tag_word[0] for tag_word in transition_matrix])
    for k in range(1, len(sentence)):
        S[k] = states

    for k in range(1, len(sentence)):
        for i in range(k - 1):
            max_pi = 0
            max_v = 0
            max_k = 0
            max_u = 0
            for j in range(k):
                u = S[i]
                v = S[j]
                if pi[(k - 1, u) * transition_matrix[u][v] * emission_matrix[v][sentence[k]]] > max_pi:
                    max_pi = pi[(k - 1, u) * transition_matrix[u][v] * emission_matrix[v][sentence[k]]]
                    max_u = u
                    max_v = v
                    max_k = k
            bp[(max_k, max_v)] = max_u
            pi[(max_k, max_v)] = max_pi


    return []

# for tag in states:
#     viterbi_table[0][tag] = {PROB: initial_prob[tag] * emission_matrix[tag], PREV: None}


# viterbi_table = [{}]
#
# observation = set([word_tag[0] for word_tag in emission_matrix])
