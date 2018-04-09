import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import re
from collections import defaultdict
from data.scorer import score_submission, print_confusion_matrix, score_defaults, SCORE_REPORT
from nltk import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm

# NOTE: Requires these following libraries:
# 'pip install matplotlib'  (Plotting)
# 'pip install numpy'  (Maths, Arrays/Matrices)
# 'pip install nltk'  (Pre-processing)
# 'pip install tqdm'  (Loading bars)


# Load training data from CSV
f_bodies = open('data/train_bodies.csv', 'r', encoding='utf-8')
csv_bodies = csv.DictReader(f_bodies)
bodies = []
for row in csv_bodies:
    body_id = int(row['Body ID'])
    if (body_id + 1) > len(bodies):
        bodies += [None] * (body_id + 1 - len(bodies))
    bodies[body_id] = row['articleBody']
f_bodies.close()
body_inverse_index = {bodies[i]: i for i in range(len(bodies))}

all_unrelated, all_discuss, all_agree, all_disagree = [], [], [], []  # each article = (headline, body, stance)

f_stances = open('data/train_stances.csv', 'r', encoding='utf-8')
csv_stances = csv.DictReader(f_stances)
for row in csv_stances:
    body = bodies[int(row['Body ID'])]
    if row['Stance'] == 'unrelated':
        all_unrelated.append((row['Headline'], body, row['Stance']))
    elif row['Stance'] == 'discuss':
        all_discuss.append((row['Headline'], body, row['Stance']))
    elif row['Stance'] == 'agree':
        all_agree.append((row['Headline'], body, row['Stance']))
    elif row['Stance'] == 'disagree':
        all_disagree.append((row['Headline'], body, row['Stance']))
f_stances.close()


# Split to train and validation 9:1
print('\tUnrltd\tDiscuss\t Agree\tDisagree')
print('All\t', len(all_unrelated), '\t', len(all_discuss), '\t', len(all_agree), '\t', len(all_disagree))

train_unrelated = all_unrelated[:len(all_unrelated) * 9 // 10]
train_discuss = all_discuss[:len(all_discuss) * 9 // 10]
train_agree = all_agree[:len(all_agree) * 9 // 10]
train_disagree = all_disagree[:len(all_disagree) * 9 // 10]

val_unrelated = all_unrelated[len(all_unrelated) * 9 // 10:]
val_discuss = all_discuss[len(all_discuss) * 9 // 10:]
val_agree = all_agree[len(all_agree) * 9 // 10:]
val_disagree = all_disagree[len(all_disagree) * 9 // 10:]

print('Train\t', len(train_unrelated), '\t', len(train_discuss), '\t', len(train_agree), '\t', len(train_disagree))
print('Valid.\t', len(val_unrelated), '\t', len(val_discuss), '\t', len(val_agree), '\t', len(val_disagree))

train_all = train_unrelated + train_discuss + train_agree + train_disagree  # each article = (headline, body, stance)
random.Random(0).shuffle(train_all)
train_all = np.array(train_all)

val_all = val_unrelated + val_discuss + val_agree + val_disagree
random.Random(0).shuffle(val_all)
val_all = np.array(val_all)

print('Train (Total)', train_all.shape, '\tValidation (Total)', val_all.shape)
print(np.count_nonzero(train_all[:, 2] == 'unrelated'), '\t',
      np.count_nonzero(train_all[:, 2] == 'discuss'), '\t',
      np.count_nonzero(train_all[:, 2] == 'agree'), '\t',
      np.count_nonzero(train_all[:, 2] == 'disagree'))
print(np.count_nonzero(val_all[:, 2] == 'unrelated'), '\t',
      np.count_nonzero(val_all[:, 2] == 'discuss'), '\t',
      np.count_nonzero(val_all[:, 2] == 'agree'), '\t',
      np.count_nonzero(val_all[:, 2] == 'disagree'))


# Tokenise text
pattern = re.compile("[^a-zA-Z0-9 ]+")  # strip punctuation, symbols, etc.
stop_words = set(stopwords.words('english'))


def tokenise(text):
    text = pattern.sub('', text.replace('\n', ' ').replace('-', ' ').lower())
    text = [word for word in word_tokenize(text) if word not in stop_words]
    return text


print('\nTokenise text')
for i in range(5):
    print(train_all[i, 0], tokenise(train_all[i, 0]))


# Compute term-frequency of words in documents
def doc_to_tf(text, ngram=1):
    words = tokenise(text)
    ret = defaultdict(float)
    for i in range(len(words)):
        for j in range(1, ngram + 1):
            if i - j < 0:
                break
            word = [words[i - k] for k in range(j)]
            ret[word[0] if ngram == 1 else tuple(word)] += 1.0
    return ret


print('\nTerm frequencies')
for i in range(5):
    print(train_all[i, 0], doc_to_tf(train_all[i, 0]))
print(train_all[0, 0], doc_to_tf(train_all[0, 0], ngram=2))


# Build corpus of article bodies and headlines in training dataset
corpus = np.r_[train_all[:, 1], train_all[:, 0]]  # 0 to 44973 are bodies, 44974 to 89943 are headlines


# Learn idf of every word in the corpus
print('\nDF')
df = defaultdict(float)
for doc in tqdm(corpus):
    words = tokenise(doc)
    seen = set()
    for word in words:
        if word not in seen:
            df[word] += 1.0
            seen.add(word)
print(list(df.items())[:10])

print('IDF')
idf = defaultdict(float)
for word, val in tqdm(df.items()):
    idf[word] = np.log((1.0 + corpus.shape[0]) / (1.0 + val)) + 1.0  # smoothed idf
print(list(idf.items())[:10])


# Load GLoVe word vectors
f_glove = open("data/glove.6B.50d.txt", "rb")  # download from https://nlp.stanford.edu/projects/glove/
glove_vectors = {}
for line in tqdm(f_glove):
    glove_vectors[str(line.split()[0]).split("'")[1]] = np.array(list(map(float, line.split()[1:])))

print('GLoVe vectors')
print(glove_vectors['glove'])


# Convert a document to GloVe vectors, by computing tf-idf of each word * GLoVe of word / total tf-idf for document
def doc_to_glove(doc):
    doc_tf = doc_to_tf(doc)
    doc_tf_idf = defaultdict(float)
    for word, tf in doc_tf.items():
        doc_tf_idf[word] = tf * idf[word]

    doc_vector = np.zeros(glove_vectors['glove'].shape[0])
    if np.sum(list(doc_tf_idf.values())) == 0.0:  # edge case: document is empty
        return doc_vector

    for word, tf_idf in doc_tf_idf.items():
        if word in glove_vectors:
            doc_vector += glove_vectors[word] * tf_idf
    doc_vector /= np.sum(list(doc_tf_idf.values()))
    return doc_vector


for i in range(2):
    print(train_all[i, 0], doc_to_glove(train_all[i, 0]))


# Compute cosine similarity of GLoVe vectors for all headline-body pairs
def dot_product(vec1, vec2):
    sigma = 0.0
    for i in range(vec1.shape[0]):  # assume vec1 and vec2 has same shape
        sigma += vec1[i] * vec2[i]
    return sigma


def magnitude(vec):
    return np.sqrt(np.sum(np.square(vec)))


def cosine_similarity(doc):
    headline_vector = doc_to_glove(doc[0])
    body_vector = doc_to_glove(doc[1])

    if magnitude(headline_vector) == 0.0 or magnitude(body_vector) == 0.0:  # edge case: document is empty
        return 0.0

    return dot_product(headline_vector, body_vector) / (magnitude(headline_vector) * magnitude(body_vector))


print('\nCosine similarity of word vectors')
for i in range(10):
    # unrelated should have lower than rest
    print(cosine_similarity(train_all[i]), train_all[i, 2])
print(cosine_similarity(train_all[27069]), tokenise(train_all[27069, 0]), tokenise(train_all[27069, 1]))  # edge case


# Compute the KL-Divergence of language model (LM) representations of the headline and the body
def divergence(lm1, lm2):
    sigma = 0.0
    for i in range(lm1.shape[0]):  # assume lm1 and lm2 has same shape
        sigma += lm1[i] * np.log(lm1[i] / lm2[i])
    return sigma


def kl_divergence(doc, eps=0.1):
    # Convert headline and body to 1-gram representations
    tf_headline = doc_to_tf(doc[0])
    tf_body = doc_to_tf(doc[1])

    # Convert dictionary tf representations to vectors (make sure columns match to the same word)
    words = set(tf_headline.keys()).union(set(tf_body.keys()))
    vec_headline, vec_body = np.zeros(len(words)), np.zeros(len(words))
    i = 0
    for word in words:
        vec_headline[i] += tf_headline[word]
        vec_body[i] = tf_body[word]
        i += 1

    # Compute a simple 1-gram language model of headline and body
    lm_headline = vec_headline + eps
    lm_headline /= np.sum(lm_headline)
    lm_body = vec_body + eps
    lm_body /= np.sum(lm_body)

    # Return KL-divergence of both language models
    return divergence(lm_headline, lm_body)


print('\nKL-divergence of language models')
for i in range(10):
    # unrelated should have higher than rest
    print(kl_divergence(train_all[i]), train_all[i, 2])
print(kl_divergence(train_all[27069]), tokenise(train_all[27069, 0]), tokenise(train_all[27069, 1]))  # edge case


# Other feature 1
def ngram_overlap(doc):
    # Returns how many times n-grams (up to 3-gram) that occur in the article's headline occur on the article's body.
    tf_headline = doc_to_tf(doc[0], ngram=3)
    tf_body = doc_to_tf(doc[1], ngram=3)
    matches = 0.0
    for words in tf_headline.keys():
        if words in tf_body:
            matches += tf_body[words]
    return np.power((matches / len(tokenise(doc[1]))), 1 / np.e)  # normalise for document length


print('\n3-gram overlap')
for i in range(10):
    # unrelated should have lower than rest
    print(ngram_overlap(train_all[i]), train_all[i, 2])


# Define function to convert each document (headline, body) to feature vectors
ftrs = [cosine_similarity, kl_divergence, ngram_overlap]
def to_feature_array(doc):
    vec = np.array([0.0] * len(ftrs))
    for i in range(len(ftrs)):
        vec[i] = ftrs[i](doc)
    return vec


# Initialise X (matrix of feature vectors) for train dataset
print('\nx_train')
x_train = np.array([to_feature_array(doc) for doc in tqdm(train_all)])
print(x_train[:10])

# Define label <-> int mappings for y
label_to_int = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
int_to_label = ['agree', 'disagree', 'discuss', 'unrelated']

# Initialise Y (gold output vector) for train dataset
y_train = np.array([label_to_int[i] for i in train_all[:, 2]])
print('y_train')
print(y_train[:10])

# Check integrity of X and Y
print('Check NaN', np.where(np.isnan(x_train)))
print('Check inf', np.where(np.isfinite(x_train) == False))
print(x_train.shape)
print(y_train.shape)  # x_train.shape[0] == y_train.shape[0]


# Plot GLoVe distance vs KL-Divergence on a coloured scatter plot with different colours for each label
colours = np.array(['g', 'r', 'b', 'y'])
plt.scatter(list(x_train[:, 0]), list(x_train[:, 1]), c=colours[y_train])
plt.xlabel('Cosine Similarity of GLoVe vectors')
plt.ylabel('KL Divergence of Unigram LMs')
print([(colours[i], int_to_label[i]) for i in range(len(int_to_label))])
# plt.show()


# Initialise x (feature vectors) for validation dataset
print('x_validation')
x_val = np.array([to_feature_array(doc) for doc in tqdm(val_all)])
print(x_val[:5])


# Linear regression model
def mse(pred, gold):
    sigma = 0.0
    for i in range(pred.shape[0]):
        sigma += np.square(pred[i] - gold[i])
    return sigma / (2 * pred.shape[0])


class LinearRegression:
    def __init__(self, lrn_rate, n_iter):
        self.lrn_rate = lrn_rate
        self.n_iter = n_iter
        # self.breakpoints = set([n_iter * i // 10 for i in range(1, 11)])

    def fit(self, X, Y):
        # Learn a model y = intercept + x0*t0 + x1*t1 + x2*t2 + ... that minimises MSE. Need to optimise T
        # self.intercept = 0.0
        self.model = np.zeros(X.shape[1])  # model[0] = t0, model[1] = t1, etc.
        for it in tqdm(range(self.n_iter)):
            model_Y = self.transform(X)
            # Thetas
            for col in range(X.shape[1]):
                s = 0.0
                for row in range(X.shape[0]):
                    s += (model_Y[row] - Y[row]) * X[row, col]
                self.model[col] -= self.lrn_rate * s / X.shape[0]
                # Intercept
                # s_int = 0.0
                # for row in range(X.shape[0]):
                #     s_int += (model_Y[row] - Y[row]) * 1.0
                # self.intercept -= self.lrn_rate * s_int / X.shape[0]
                # if it + 1 in self.breakpoints:
                # print('Iteration', it+1, 'MSE:', mse(model_Y, Y))
        print('Final MSE:', mse(model_Y, Y))
        # print('Intercept:', self.intercept)
        print('Model:', self.model)

    def transform(self, X):
        # Returns a float value for each X. (Regression)
        Y = np.zeros(X.shape[0])
        for row in range(X.shape[0]):
            # s = self.intercept
            s = 0.0
            for col in range(X.shape[1]):
                s += self.model[col] * X[row, col]
            Y[row] = s
        return Y

    def predict(self, X):
        # Uses results of transform() for binary classification. For testing only, use OneVAllClassifier otherwise.
        Y = self.transform(X)
        Y = np.array([(1 if i > 0.5 else 0) for i in Y])
        return Y


# Logistic regression functions
def sigmoid(Y):
    return 1 / (1 + np.exp(Y * -1))


def logistic_cost(pred, gold):
    sigma = 0.0
    for i in range(pred.shape[0]):
        if gold[i] == 1:
            sigma -= np.log(pred[i])
        elif gold[i] == 0:
            sigma -= np.log(1 - pred[i])
    return sigma / pred.shape[0]


# Logistic regression model
class LogisticRegression:
    def __init__(self, lrn_rate, n_iter):
        self.lrn_rate = lrn_rate
        self.n_iter = n_iter
        # self.breakpoints = set([n_iter * i // 10 for i in range(1, 11)])

    def fit(self, X, Y):
        # Learn a model y = x0*t0 + x1*t1 + x2*t2 + ... that minimises MSE. Need to optimise T
        self.model = np.zeros(X.shape[1])  # model[0] = t0, model[1] = t1, etc.
        for it in tqdm(range(self.n_iter)):
            model_Y = self.transform(X)
            for col in range(X.shape[1]):
                s = 0.0
                for row in range(X.shape[0]):
                    s += (model_Y[row] - Y[row]) * X[row, col]
                self.model[col] -= self.lrn_rate * s / X.shape[0]
                # if it + 1 in self.breakpoints:
                # print('Iteration', it+1, 'loss:', logistic_cost(model_Y, Y))
        print('Final loss:', logistic_cost(model_Y, Y))
        print('Model:', self.model)

    def transform(self, X):
        # Returns a float value for each X. (Regression)
        Y = np.zeros(X.shape[0])
        for row in range(X.shape[0]):
            s = 0.0
            for col in range(X.shape[1]):
                s += self.model[col] * X[row, col]
            Y[row] = s
        return sigmoid(Y)

    def predict(self, X):
        # Uses results of transform() for binary classification. For testing only, use OneVAllClassifier otherwise.
        Y = self.transform(X)
        Y = np.array([(1 if i > 0.5 else 0) for i in Y])
        return Y


# To use linear/logistic regression models to classify multiple classes
class OneVAllClassifier:
    def __init__(self, regression, **params):
        self.regression = regression
        self.params = params

    def fit(self, X, Y):
        # Learn a model for each parameter.
        self.categories = np.unique(Y)
        self.models = {}
        for cat in self.categories:
            ova_Y = np.array([(1 if i == cat else 0) for i in Y])
            model = self.regression(**self.params)
            model.fit(X, ova_Y)
            self.models[cat] = model
            print(int_to_label[cat])

    def predict(self, X):
        # Predicts y for each different model learned, and returns the category related to the model with highest score.
        vals = {}
        for cat, model in self.models.items():
            vals[cat] = model.transform(X)
        Y = np.zeros(X.shape[0], dtype=np.int)
        for row in range(X.shape[0]):
            max_val, max_cat = -math.inf, -math.inf
            for cat, val in vals.items():
                if val[row] > max_val:
                    max_val, max_cat = val[row], cat
            Y[row] = max_cat
        return Y


# Train the linear regression & One-V-All classifier models on the train set
print('\nTrain Models: Linear Regression')
clf = OneVAllClassifier(LinearRegression, lrn_rate=0.1, n_iter=1000)
clf.fit(x_train, y_train)

# Predict y for validation set
print('\nRESULTS: VALIDATION SET LINEAR REGRESSION')
y_pred = clf.predict(x_val)
print(y_pred[:5])
predicted = np.array([int_to_label[i] for i in y_pred])
print(predicted[:5])
print(val_all[:, 2][:5])

# Prepare validation dataset format for score_submission in scorer.py
body_ids = [str(body_inverse_index[body]) for body in val_all[:, 1]]
pred_for_cm = np.array([{'Headline': val_all[i, 0], 'Body ID': body_ids[i], 'Stance': predicted[i]} for i in range(len(val_all))])
gold_for_cm = np.array([{'Headline': val_all[i, 0], 'Body ID': body_ids[i], 'Stance': val_all[i, 2]} for i in range(len(val_all))])

# Score using scorer.py (provided in https://github.com/FakeNewsChallenge/fnc-1) on VALIDATION set:
test_score, cm = score_submission(gold_for_cm, pred_for_cm)
null_score, max_score = score_defaults(gold_for_cm)
print_confusion_matrix(cm)
print(SCORE_REPORT.format(max_score, null_score, test_score))


# Predict y for validation set using logistic regression instead of linear regression, and compare results of scorer.py
print('\nTrain Models: Logistic Regression')
clf_logistic = OneVAllClassifier(LogisticRegression, lrn_rate=0.1, n_iter=1000)
clf_logistic.fit(x_train, y_train)

print('\nRESULTS: VALIDATION SET LOGISTIC REGRESSION')
y_pred = clf_logistic.predict(x_val)
predicted = np.array([int_to_label[i] for i in y_pred])

body_ids = [str(body_inverse_index[body]) for body in val_all[:, 1]]
pred_for_cm = np.array([{'Headline': val_all[i, 0], 'Body ID': body_ids[i], 'Stance': predicted[i]}
                        for i in range(len(val_all))])
gold_for_cm = np.array([{'Headline': val_all[i, 0], 'Body ID': body_ids[i], 'Stance': val_all[i, 2]}
                        for i in range(len(val_all))])

test_score, cm = score_submission(gold_for_cm, pred_for_cm)
null_score, max_score = score_defaults(gold_for_cm)
print_confusion_matrix(cm)
print(SCORE_REPORT.format(max_score, null_score, test_score))
# linear regression performs better, so that model is chosen for the test set


# Load test data from CSV
f_tbodies = open('data/competition_test_bodies.csv', 'r', encoding='utf-8')
csv_tbodies = csv.DictReader(f_tbodies)
tbodies = []
for row in csv_tbodies:
    body_id = int(row['Body ID'])
    if (body_id + 1) > len(tbodies):
        tbodies += [None] * (body_id + 1 - len(tbodies))
    tbodies[body_id] = row['articleBody']
f_tbodies.close()
tbody_inverse_index = {tbodies[i]: i for i in range(len(tbodies))}

test_all = []  # each article = (headline, body, stance)

f_tstances = open('data/competition_test_stances.csv', 'r', encoding='utf-8')
csv_tstances = csv.DictReader(f_tstances)
for row in csv_tstances:
    body = tbodies[int(row['Body ID'])]
    test_all.append((row['Headline'], body, row['Stance']))
f_tstances.close()

# Initialise x (feature vectors) and y for test dataset
x_test = np.array([to_feature_array(doc) for doc in tqdm(test_all)])
print('\nx_test')
print(x_test[:5])

# Predict y for test set
print('\nRESULTS: TEST SET LINEAR REGRESSION')
y_test = clf.predict(x_test)
print(y_pred[:5])
pred_test = np.array([int_to_label[i] for i in y_test])
print(pred_test[:5])

# Prepare test dataset format for score_submission in scorer.py
test_body_ids = [str(tbody_inverse_index[test_all[i][1]]) for i in range(len(test_all))]
test_pred_for_cm = np.array([{'Headline': test_all[i][0], 'Body ID': test_body_ids[i], 'Stance': pred_test[i]}
                             for i in range(len(test_all))])
test_gold_for_cm = np.array([{'Headline': test_all[i][0], 'Body ID': test_body_ids[i], 'Stance': test_all[i][2]}
                             for i in range(len(test_all))])

# Score using scorer.py (provided in https://github.com/FakeNewsChallenge/fnc-1) on TEST set:
test_score, cm = score_submission(test_gold_for_cm, test_pred_for_cm)
null_score, max_score = score_defaults(test_gold_for_cm)
print_confusion_matrix(cm)
print(SCORE_REPORT.format(max_score, null_score, test_score))
