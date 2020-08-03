import re
import string

import pandas as pd, numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# 读取数据
from sklearn.metrics import classification_report
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
submission_file = pd.read_csv('sample_submission.csv')

# 检查 train set 中的数据构成
train.head()
print(train['comment_text'][0])
print(train['comment_text'][2])

# comment lines 长度分析
lens = train.comment_text.str.len()
print('comment lines analyze')
print(f'Average length = {lens.mean()}\nStandard deviation = {lens.std()}\nMax length = {lens.max()}')

# 列出所有的label，并把 ‘没有任何一个label的comment’的类型定义为 none
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train['none'] = 1 - train[label_cols].max(axis=1)
# 统计label类别
print('label description')
train_stat = train.describe()
train_stat.to_csv("label_stat.csv", index=False)
# print(a, file=f)

# 原始数据中有几段空白行，把他们去除
sentence = 'comment_text'
# train sentence中去除
train[sentence].fillna("unknown", inplace=True)
# test sentence 中去除
test[sentence].fillna("unknown", inplace=True)
test_id = test['id']


# def predict_and_test(model, X_test_bag_of_words):
#     predicted_y = model.predict(X_test_bag_of_words)
#     # print(testing_y, predicted_y)
#     # print(model.predict_proba(X_test_bag_of_words))
#     print(classification_report(testing_y, predicted_y, zero_division=0))


## 构建模型，使用NB-SVM模型
# use f' to format the string
# string.punctuation is a readymade-string of common punctuation marks.
# re.compile(f'...') compiles the expression([{string.punctuation}...]) into pattern objects
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')


# re_tok.sub(r' \1 ', s) finds all the resulting matching-punctuations and
# adds a prefix and suffix of white-space to those matching patterns.
# Lastly, split() call tokenizes resulting string into an array of individual
# words and punctuation marks.
def tokenize(s): return re_tok.sub(r' \1 ', s).split()


n = train.shape[0]

# use tokenizer to mark the the weird characters on their own
# use min_df and max_df to ignore some words that appear too frequently
#   or infrequently.
# use_idf: Start inverse-document-frequency to recalculate weight
# smooth_idf: Smooth the idf weight by adding 1 to the document frequency.
#   To prevent division by zero, add an additional document.
# sublinear_tf: Apply linear scaling TF, for example, use 1 + log(tf) to cover tf

tfidfvec = TfidfVectorizer(ngram_range=(1, 2), tokenizer=tokenize,
                           min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                           smooth_idf=1, sublinear_tf=1)
trn_term_doc = tfidfvec.fit_transform(train[sentence])
test_term_doc = tfidfvec.transform(test[sentence])


class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            # find the y==y_i in x, and sum by column(axis = 0)
            p = x[y == y_i].sum(0)
            return (p + 1) / ((y == y_i).sum() + 1)

        self._r = sparse.csr_matrix(np.log(pr(x, 1, y) / pr(x, 0, y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y)
        return self


# model = NbSvmClassifier(C=4, dual=True, n_jobs=-1).fit(trn_term_doc, training_labels)


# for i, j in enumerate(label_cols):
#     print('fit', j)
#     model = NbSvmClassifier(C=4, dual=False, n_jobs=-1).fit(trn_term_doc, train[j])
#
# predict_y = model.predict(test_term_doc)
#
# for i in range(len(test)):
#     print(test_id[i], predict_y[i])
# basic naive bayes feature equation:
# def pr(y_i, y):
#     p = x[y == y_i].sum(0)
#     return (p + 1) / ((y == y_i).sum() + 1)
#
#
# x = trn_term_doc
# test_x = test_term_doc
#
#
# # Fit a model for one dependent at a time
# def get_mdl(y):
#     y = y.values
#     r = np.log(pr(1, y) / pr(0, y))
#     m = LogisticRegression(C=4, dual=False)
#     x_nb = x.multiply(r)
#     return m.fit(x_nb, y), r
#


# use np.zeros to create a zero array
prediction_array = np.zeros((len(test), len(label_cols)))
for i, j in enumerate(label_cols):
    print('fit', j)
    model = NbSvmClassifier(C=4, dual=False, n_jobs=-1).fit(trn_term_doc, train[j])
    prediction_array[:, i] = model.predict_proba(test_term_doc)[:, 1]

submid = pd.DataFrame({'id': submission_file["id"]})
submission = pd.concat([submid, pd.DataFrame(prediction_array, columns=label_cols)], axis=1)
submission.to_csv('submission.csv', index=False)


