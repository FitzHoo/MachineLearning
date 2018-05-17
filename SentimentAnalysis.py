import numpy as np
import pyprind
import pandas as pd
import os
os.chdir('/Users/huliangyong/Docs/MachineLearning')

# pbar = pyprind.ProgBar(50000)
# labels = {'pos': 1, 'neg': 0}
# df = pd.DataFrame()
# for s in ('test', 'train'):
#     for l in ("pos", 'neg'):
#         path = './aclImdb/{}/{}'.format(s, l)
#         for file in os.listdir(path):
#             with open(os.path.join(path, file), 'r') as infile:
#                 txt = infile.read()
#             df = df.append([[txt, labels[l]]], ignore_index=True)
#             pbar.update()
# df.columns = ['review', 'sentiment']
#
# np.random.seed(0)
# df = df.reindex(np.random.permutation(df.index))
# df.to_csv('./movie_data.csv', index=False)
# print(df.head())
df = pd.read_csv('./movie_data.csv')

from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
docs = np.array([
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining and the weather is sweet'
])

bag = count.fit_transform(docs)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
np.set_printoptions(precision=2)
print(tfidf.fit_transform((count.fit_transform(docs))).toarray())

# 使用正则表达式处理文本
import re


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ''.join(emoticons).replace('-', '')
    return text

df['review'] = df['review'].apply(preprocessor)

def tokenizer(text):
    return text.split()

test = 'runners like running and thus they run'

from nltk.stem.porter import PorterStemmer

# 将单词恢复到原始形式
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

tokenizer_porter(test)


# from nltk import download
# download("stopwords")

from nltk.corpus import stopwords
stop = stopwords.words("english")
tt = [w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop]

# 训练用于文档分类的逻辑斯蒂回归
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['review'].values, df['sentiment'].values, test_size=0.5)

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf': [False],
               'vect__norm': [None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]}
              ]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0))])
# gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
# gs_lr_tfidf.fit(X_train, y_train)
#
# print('Best parameter set: {}'.format(gs_lr_tfidf.best_params_))
# print('CV Accuracy {:.3f}'.format(gs_lr_tfidf.best_score_))
# clf = gs_lr_tfidf.best_estimator_
# print('Test Accuracy {:.3f}'.format(clf.score(X_test, y_test)))


def tokenizer_(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ''.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

# 定义一个生成器函数
def stream_docs(path):
    with open(path , 'r') as csv:
        next(csv)  # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

next(stream_docs(path='./movie_data.csv'))

def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None

    return docs, y


from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

vect = HashingVectorizer(decode_error='ignore', n_features=2**21, preprocessor=None, tokenizer=tokenizer_)
clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
doc_stream = stream_docs(path='./movie_data.csv')

pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print("Accuracy: {:.3f}".format(clf.score(X_test, y_test)))
clf = clf.partial_fit(X_test, y_test)


