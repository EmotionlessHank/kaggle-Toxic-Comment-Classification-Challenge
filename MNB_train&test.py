import re
import string

import pandas as pd, numpy as np
from nltk import PorterStemmer
from nltk.corpus import stopwords
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# 读取数据
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_labels = pd.read_csv('test_labels.csv')
submission_file = pd.read_csv('sample_submission.csv')

# 列出所有的label，并把 ‘没有任何一个label的comment’的类型定义为 none
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1 - train[label_cols].max(axis=1)

# 原始数据中有几段空白行，把他们去除
sentence = 'comment_text'
# train sentence中去除
train[sentence].fillna("unknown", inplace=True)
# test sentence 中去除
test[sentence].fillna("unknown", inplace=True)
test_id = test['id']


def stop_words_remove(sentence):
    stop_words = set(stopwords.words('english'))
    words = sentence.split(' ')
    filtered_words = [w for w in words if w not in stop_words]
    remove_stop_sentence = ' '.join(w for w in filtered_words)
    return remove_stop_sentence


def data_preprocessing(sentence):
    result = []
    for i in sentence:
        remove_special_char = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", " ",i)
        remove_stop_words = stop_words_remove(remove_special_char)
        result.append(remove_stop_words)
    return result


# remove '\\n'
train['comment_text'] = train['comment_text'].map(lambda x: re.sub('\\n', ' ', str(x)))

# remove any text starting with User...
train['comment_text'] = train['comment_text'].map(lambda x: re.sub("\[\[User.*", '', str(x)))

# remove IP addresses or user IDs
train['comment_text'] = train['comment_text'].map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", '', str(x)))

# remove http links in the text
train['comment_text'] = train['comment_text'].map(lambda x: re.sub("(http://.*?\s)|(http://.*)", '', str(x)))

# remove '\\n'
test['comment_text'] = test['comment_text'].map(lambda x: re.sub('\\n', ' ', str(x)))

# remove any text starting with User...
test['comment_text'] = test['comment_text'].map(lambda x: re.sub("\[\[User.*", '', str(x)))

# remove IP addresses or user IDs
test['comment_text'] = test['comment_text'].map(
    lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", '', str(x)))

# remove http links in the text
test['comment_text'] = test['comment_text'].map(lambda x: re.sub("(http://.*?\s)|(http://.*)", '', str(x)))

x = train['comment_text']
y = train.iloc[:, 2:8]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=13)


valid_train_sentence, valid_test_sentence = data_preprocessing(X_train), data_preprocessing(X_test)



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


counter = CountVectorizer(token_pattern=token, ngram_range=(1, 2), tokenizer=tokenize,
                          min_df=3, max_df=0.9, strip_accents='unicode')
tfidfvec = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize,
                           min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                           smooth_idf=1, sublinear_tf=1, max_features=3750)
X_train_bag_of_words = tfidfvec.fit_transform(valid_train_sentence)
X_test_bag_of_words = tfidfvec.transform(valid_test_sentence)

clf = MultinomialNB()
clf2 = BernoulliNB()
# set the min samples leaf as 1% of training sets.
min_value = int(0.01 * len(train[sentence]))
clf3 = tree.DecisionTreeClassifier(min_samples_leaf=min_value, criterion='entropy', random_state=0)

# fit the model then test
prediction_array = np.zeros((len(test), len(label_cols)))
for i, j in enumerate(label_cols):
    print('fit', j)
    train_target = y_train[j]
    test_target = y_test[j]
    model = clf.fit(X_train_bag_of_words, train_target)
    predicted_y = model.predict(X_test_bag_of_words)
    report = classification_report(test_target, predicted_y,output_dict=True)

    print(classification_report(test_target, predicted_y))
    df = pd.DataFrame(report).transpose()
    df.to_csv(f"classification_result_{j}.csv", index=True)
    # prediction_array[:, i] = model.predict_proba(X_test_bag_of_words)[:, 1]

# # output the submission
# submid = pd.DataFrame({'id': submission_file["id"]})
# submission = pd.concat([submid, pd.DataFrame(prediction_array, columns=label_cols)], axis=1)
# submission.to_csv('submission_MNB_tfidfvec_SWSTEM.csv', index=False)
#
# # my_report = classification_report(testing_y, predict_y, zero_division=0)
# for class_name in enumerate(label_cols):
#     train_target = prediction_array[class_name]
#     test_target = prediction_array[class_name]