from __future__ import division
import cPickle
import numpy as np
import pylab as pl
from pandas import DataFrame
from arsenal.alphabet import Alphabet
from arsenal.iterview import progress
from arsenal.terminal import colors
from collections import Counter, defaultdict
from grafl.test import make_model_func
from grafl.dataset.edge_dataset import BWD_dataset

np.set_printoptions(precision=4)

L = {
    0: 'coordinate',
    1: 'hypernym',
    2: 'hyponym',
}

A = Alphabet()
A.map([x.strip().split()[1] for i, x in enumerate(file('res/bowman_wordnet_longer_shuffled_synset_relations.map')) if i > 2])

tst = BWD_dataset('test').data
trn = BWD_dataset('train').data
trn_x = trn[0]
trn_y = trn[1]
seen = set(trn_x.flatten()) | set(trn_y.flatten())

X,Y,_ = tst

X = list(A.lookup_many(X.flatten()))
Y = list(A.lookup_many(Y.flatten()))
#D = np.array([X,Y,L.flatten()]).T

model_file = 'res/experiments/BWD-projection-Softmax_best.pkl'
#model_file = '/home/timv/Downloads/BWD-projection-identity_sub_glue-Softmax.pkl'
model_func = make_model_func(cPickle.load(open(model_file, 'rb')))
(x_left, x_right, y_true) = tst

y_true = y_true.flatten()
y_dist = model_func((x_left, x_right))
y_hat = y_dist.argmax(axis=1)


print 'before unseen filter acc: %g' % (y_hat == y_true).mean()


unseen = 0
unseen_err = 0
err = 0

#correct_unseen = 0
#correct = 0

for t,p,x,y in sorted(zip(y_true,y_hat,X,Y)):
    ux = A[x] not in seen
    uy = A[y] not in seen
    if ux or uy:
        unseen += 1
        continue       # filter-out unseen test examples
    if p != t:
        err += 1
#        print x, y, colors.red % L[p], colors.green % L[t], 'unseen(x)' if ux else '', 'unseen(y)' if uy else ''
#    else:
#        if ux or uy:
#            correct_unseen += 1
#        correct += 1


print 'p(err|seen) = %s' % progress(err, len(y_true))
print 'p(unseen)   = %s' % progress(unseen, len(y_true))

#print 'correct unseen = %s' % progress(correct_unseen, correct)


"""TODO

 - How transitive is the learned relation?

"""




"""
Confusion matrix
================

C[i,j] = "pred i, true j"
"""

C = np.zeros((3,3))
for t,p,x,y in sorted(zip(y_true,y_hat,X,Y)):
    if A[x] not in seen or A[y] not in seen:
        continue       # filter-out unseen test examples
    C[p,t] += 1

C_true = C.sum(axis=0)

print C
print C_true
print C/C_true

"""
Word counts
"""

cnt = Counter(list(A.lookup_many(trn_x.flatten())) + list(A.lookup_many(trn_y.flatten())))

print
print 'most common'
print '==========='
for k,v in cnt.most_common(20):
    print '%3s %s' % (v, k)

print 'total train:', len(trn_x)


"""
Error rates and entity frequency in the training data.

 - How many times have we seen the word in train v. error?

   Generalizes these questions:
    - How many of these errors are wrt organism?

      We never make errors on 'organism'

    - How many are unseen words?

"""

word_err = {k: [0,0,[],[]] for k in cnt}

data = []
for t,p,x,y in sorted(zip(y_true,y_hat,X,Y)):
    if A[x] not in seen or A[y] not in seen:
        continue       # filter-out unseen test examples

    word_err[x][0] += 1
    word_err[y][0] += 1

    if p != t:
        word_err[x][1] += 1
        word_err[y][1] += 1
        word_err[y][2].append(x)
        word_err[x][2].append(y)
    else:
        word_err[y][3].append(x)
        word_err[x][3].append(y)

    data.append({'min': min(cnt[x], cnt[y]),
                 'max': max(cnt[x], cnt[y]),
                 'avg': (cnt[x]+cnt[y])/2,
                 'cntx': cnt[x],
                 'cnty': cnt[y],
                 'err': t!=p})

df = DataFrame(data)

if 1:
    pl.figure()
    pl.hist(list(df[df.err]['min']), bins=16)
    pl.title('Histogram of min(cnt[x], cnt[y]) given error')

if 1:
    pl.figure()
    pl.hist(list(df[~df.err]['min']), bins=16)
    pl.title('Histogram of min(cnt[x], cnt[y]) given correct')


print
print 'most common and the errors'
print '=========================='
for k,v in cnt.most_common(30):
    if word_err[k][0] == 0:
        continue
#    print '%3s %30s %s: %s | %s' % (v, k, progress(word_err[k][1], word_err[k][0]),
#                                    ' '.join(word_err[k][3]),
#                                    ' '.join(word_err[k][2]))
    print '%3s %30s %s' % (v, k, progress(word_err[k][1], word_err[k][0]))
