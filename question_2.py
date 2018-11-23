import nltk
from nltk.corpus import brown
import copy

# nltk.download('brown')
news_tagged = brown.tagged_sents(categories='news')  # Download all the data
train_set = news_tagged._pieces[:(int)(len(news_tagged._pieces) * 0.9)]
test_set = news_tagged._pieces[int(len(news_tagged._pieces) * 0.9):]
# print(len(train_set))
# print(len(test_set))


def get_train_set():
    return copy.deepcopy(train_set)


def get_test_set():
    return copy.deepcopy(test_set)


def most_likely_tag():
    my_train_set = get_train_set()
    for file in my_train_set:
        for line in file:
            for couple in line:
                print(couple)

most_likely_tag()
