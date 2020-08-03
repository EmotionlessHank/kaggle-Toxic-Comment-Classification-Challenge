import re
import string

import sklearn
from  nlppreprocess import NLP
import pandas as pd, numpy as np
from nltk.corpus import stopwords
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

# read the train and test data
train = pd.read_csv('train.csv')
train = sklearn.utils.shuffle(train)
test = pd.read_csv('test.csv')
# test = sklearn.utils.shuffle(test)
test_labels = pd.read_csv('test_labels.csv')
submission_file = pd.read_csv('sample_submission.csv')

# list all the labels and store them in a list.
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
# create a new label called 'none' which is means that the comment dont have any
# toxic content
train['none'] = 1 - train[label_cols].max(axis=1)

# remove the blank lines from raw train and test sentence
sentence = 'comment_text'
# train sentence中去除
train[sentence].fillna("unknown", inplace=True)
# test sentence 中去除
test[sentence].fillna("unknown", inplace=True)
test_id = test['id']


# create a func to remove the stopwords
def stop_words_remove(sentence):
    stop_words = set(stopwords.words('english'))
    words = sentence.split(' ')
    filtered_words = [w for w in words if w not in stop_words]
    remove_stop_sentence = ' '.join(w for w in filtered_words)
    return remove_stop_sentence


# create a func to cleaning the data
def data_preprocessing(sentence):
    result = []
    for i in sentence:
        remove_special_char = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", " ", i)
        remove_backslash_n = re.sub('\\n', ' ', remove_special_char)
        remove_Startwith_User = re.sub("\[\[User.*", '', remove_backslash_n)
        remove_IP = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", '', remove_Startwith_User)
        remove_stop_words = stop_words_remove(remove_IP)
        # stemming_sentence = stemming_words(remove_stop_words)
        result.append(remove_stop_words)
    return result
# nlp = NLP()
# train[sentence] = train[sentence].apply(nlp.process)
# test[sentence] = test[sentence].apply(nlp.process)
valid_train_sentence, valid_test_sentence = data_preprocessing(train[sentence]), data_preprocessing(test[sentence])

token = r'[#@_$%\w\d]{2,}'
# use f' to format the string
# string.punctuation is a readymade-string of common punctuation marks.
# re.compile(f'...') compiles the expression([{string.punctuation}...]) into pattern objects
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')


# re_tok.sub(r' \1 ', s) finds all the resulting matching-punctuations and
# adds a prefix and suffix of white-space to those matching patterns.
# Lastly, split() call tokenizes resulting string into an array of individual
# words and punctuation marks.
def tokenize(s): return re_tok.sub(r' \1 ', s).split()


counter = CountVectorizer(token_pattern=token, ngram_range=(1, 3), tokenizer=tokenize,
                          min_df=3, max_df=0.9, strip_accents='unicode')

# use tfidf vectorizer to create words table
tfidfvec = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize,
                           min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                           smooth_idf=1, sublinear_tf=1, max_features=30000)
X_train_bag_of_words = tfidfvec.fit_transform(valid_train_sentence)
X_test_bag_of_words = tfidfvec.transform(valid_test_sentence)

# use MNB to fit the model
clf = MultinomialNB()
clf2 = BernoulliNB()

# fit the model then test
prediction_array = np.zeros((len(test), len(label_cols)))
for i, j in enumerate(label_cols):
    print('fit', j)
    model = clf.fit(X_train_bag_of_words, train[j])
    predicted_y = model.predict(X_test_bag_of_words)
    # print(classification_report(test_y[j], predicted_y))
    prediction_array[:, i] = model.predict_proba(X_test_bag_of_words)[:, 1]

# output the submission to the required format
submid = pd.DataFrame({'id': submission_file["id"]})
submission = pd.concat([submid, pd.DataFrame(prediction_array, columns=label_cols)], axis=1)
submission.to_csv('submission_MNB_30000.csv', index=False)

