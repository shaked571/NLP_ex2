import question_2


my_train_data = question_2.get_word_tag_train_full_list()
for couple in my_train_data:
    print(couple)


def viterbi_algorithm(observation, states, initial_prob, transition_matrix, emission_matrix):
    table = [{}]
    for state in states:
        table[0][state] = {"prob": initial_prob[state] * emission_matrix[state][observation[0]], "prev": None}
