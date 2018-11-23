import nltk
from nltk.corpus import brown

# nltk.download('brown')
news_tagged = brown.tagged_sents(categories='news')  # Download all the data
train_set = news_tagged._pieces[:(int)(len(news_tagged._pieces) * 0.9)]
test_set = news_tagged._pieces[int(len(news_tagged._pieces) * 0.9):]
print(len(train_set))
print(len(test_set))

# for tag_sents in brown.tagged_sents():
#     print(tag_sents)


def foo():
    print('a')

def shaked():
    print('king')

def most_likely_tag_baseline():
    for word_and_tag in train_set:
        print(word_and_tag)


