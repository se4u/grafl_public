'''
| Filename    : lib_vector_logic.py
| Description : Example implementations of vector based models used for representing predicates/terms in logic
| Author      : Pushpendre Rastogi
| Created     : Fri Jul 31 17:48:16 2015 (-0400)
| Last-Updated:
|           By:
|     Update #: 2
A large number of vector based models have been proposed for incorporating
- multi-relational knowledge
- propositional knowledge
- first order knowledge
into vector space based models.
This library contains reference implementations of some of these models.

Bowman model
1. Say we have words w1, w2
2. Find their embeddings e1, e2
3. Feed the two into a comparison function. f(e1, e2, Relation)
   a. [NN Comparator]  Just uses a v_r.f(W_r[e1, e2] + b_r)
   b. [NTN Comparator] Just uses a v_r.f(e1 . T_r . e2 + W_r[e1, e2] + b_r)
4. Then feed this into a softmax classifier
5. Regularize with L2
6. Train with AdaGrad
'''
