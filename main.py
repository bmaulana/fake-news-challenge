import csv
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Load data from CSV
f_bodies = open('data/train_bodies.csv', 'r', encoding='utf-8')
csv_bodies = csv.DictReader(f_bodies)
bodies = []
for row in csv_bodies:
    body_id = int(row['Body ID'])
    if (body_id + 1) > len(bodies):
        bodies += [None] * (body_id + 1 - len(bodies))
    bodies[body_id] = row['articleBody']
f_bodies.close()

f_stances = open('data/train_stances.csv', 'r', encoding='utf-8')
csv_stances = csv.DictReader(f_stances)
all_unrelated, all_discuss, all_agree, all_disagree = [], [], [], []
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

# Split to train and validation
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

train_all = train_unrelated + train_discuss + train_agree + train_disagree
random.Random(0).shuffle(train_all)
val_all = val_unrelated + val_discuss + val_agree + val_disagree
random.Random(0).shuffle(val_all)
print('Train (Total)', len(train_all), '\tValidation (Total)', len(val_all))

# Convert documents to tf-idf form
corpus = [i[1] for i in train_all] + [i[0] for i in train_all]  # 0 to 44973 are bodies, 44974 to 89943 are headlines
vectoriser = CountVectorizer(stop_words='english')
matrix = vectoriser.fit_transform(corpus)
transformer = TfidfTransformer(smooth_idf=False)
tf_idf = transformer.fit_transform(matrix)
# print(tf_idf[0])
col_to_word = vectoriser.get_feature_names()

# Load GLoVe word vectors
f_glove = open("data/glove.6B.50d.txt", "rb")
glove_vectors = {}
for line in f_glove:
    glove_vectors[str(line.split()[0]).split("'")[1]] = np.array(list(map(float, line.split()[1:])))
# for key, value in glove_vectors.items():
#    print(key, value)
#    break

# Convert documents to GloVe vectors, by doing tf-idf of each word * GLoVe of word / total tf-idf for document
doc_vectors = []
# print(train_all[0][1])
for row in range(tf_idf.shape[0]):  # testing: from len(train_all)
    _, cols = tf_idf[row].nonzero()
    doc_vector = np.array([0.0]*50)
    sum_tf_idf = 0
    for col in cols:
        word = col_to_word[col]
        if word in glove_vectors:
            # print(word, tf_idf[row, col], glove_vectors[word])
            doc_vector += glove_vectors[word] * tf_idf[row, col]
            sum_tf_idf += tf_idf[row, col]
    doc_vector /= sum_tf_idf
    # print(doc_vector)
    doc_vectors.append(doc_vector)
    # break
doc_vectors = np.array(doc_vectors)
print(doc_vectors.shape)

# TODO move everything to Jupyter (otherwise re-running everything to test next parts would take too long)

features = []  # indices correspond to train_all
for i in range(len(train_all)):
    print(doc_vectors[i], doc_vectors[i+len(train_all)])  # body, headline
    # TODO cosine similarity of doc_vectors[i] and doc_vectors[i+len(train_all)]
    features.append({'headline_body_similarity': 0.0})
    break  # for testing, remove later

# TODO other features: KL-Divergence,
    # entity count of entities in headline in body?,
    # occurence of unigrams/bigrams/trigrams in headline in body?

# TODO Plot GLoVe distance vs KL-Divergence on a coloured scatter plot with different colours for each label

# TODO implement linear/logistic regression classifier using these features. Optimise params on validation set

# TODO test performance of classifier on test set (confusion matrix, precision, recall, F-score)

# TODO analyse importance of each feature
