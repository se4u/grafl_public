# -*- coding: utf-8 -*-
'''
| Filename    : artificial_natlog_data.py
| Description : Classes for creating artificial models that obey natural logic.
| Author      : Pushpendre Rastogi
| Created     : Fri Jul 31 14:56:22 2015 (-0400)
| Last-Updated:
|           By:
|     Update #: 52
Sam Bowman's conducted experiments on manually constructed models [1]
to test the NN/NTN models abilities to learn inferential patterns in
Natural Logic. Each model was created as follows
a. Create 80 sets by drawing from 7 entities (randomly)
b. Based on these 80 sets, one can ask 6400 pair wise questions about
   which natural logic relation can hold between the two sets. It is
   interesting to note that the natlog relations are mutually exclusive.
c. Split the 6400 formulaes, each formula is a 3-tuple, (S1, S2, Relation)
   Note, Bowman only feeds positive experiences into the net. He doesn't
   need to feed in negative experiences but that type of knowledge may
   actually be important.
d. He also had a prop logic proof system, that he needed to weed out test
   instances that could not be proved from the training instances and the axioms.
   Actually looking at this shows that there is a very strong structure that the
   axioms introduce. (Rocktaschel/Sameer's methods?)
   * The work can be done simply through forward induction by starting from
     existing knowledge and enumerating all possible theorems in the language.
     No need for backward chaining.
e. After weeding out the system's he trained the system. I wonder if adagrad etc.
   are really needed or if basic cvxopt methods could be used?
f. Now the training was done by varying the models, drawing the sets many times,
   changing the embedding sizes etc.
   [TORCH] I could wrap "Torch objects" for the training and testing.
       Basically write some code in Torch, wrap it with
       cython, expose it as a python class/object.
       The separation is maintained using.
       https://github.com/soumith/torch-ship-binaries
g. The goal is to reproduce the accuracy that bowman et.al reported of
   |     | Provable         | UnProvable       |
   | NTN | 98.1% (SE 0.67%) | 87.7% (SE 3.59%) |
   | NN  | 84.8% (SE 0.84%) | 68.7% (SE 1.58%) |
h. After that I would optimize the representations subject to constraints of symmetry, reflexivity, transitivity
   | SNo | Name     | Symm | Refl | Tran | Remarks                                                              |
   |   1 | entails  | NO   |      | ✓    | A subtype entails supertypes, More attributes entail Less attributes |
   |   2 | rentails | NO   |      | ✓    | Reverse of entails                                                   |
   |   3 | equiv    | ✓    | ✓    | ✓    | x = y                                                                |
   |   4 | altern   | ✓    |      | ✓    | x ∩ y = Φ ∧ x ∪ y ≠ D                                                |
   |   5 | negation | ✓    |      |      | x ∩ y = Φ ∧ x ∪ y = D                                                |
   |   6 | cover    | ✓    |      |      | x ∩ y ≠ Φ ∧ x ∪ y = D                                                |
   |   7 | indep    | ✓    |      | ✓    | (else)                                                               | > 0
   X^T R y > 0
   x^T R y > 0 and y^T R z > 0 then x^T R z > 0
   R = uu^T
i. The method to enforce these constraints would be:
   # NOTE: It is not possible to enforce transitivity without symmetry in plain bilinear models.
   #  therefore plain bilinear is only useful for equiv, altern, indep.
   #  but not for entailment.
   # NOTE: The activations that a R_i b produce actually only have to compete against other {R_i}
   # The constraints in my mind make sure that the activation of (a R_i b = b R_i a)
   # but they don't say how I would perform against (b R_i a) for other {R_i}
   # NOTE: entailment/rentailment are almost anti-symmetric in their nature.
   # a entails b
   # ==> a) b rentails a
   # ==> b) b "does not entail" a
   | Model                    | Symm                                   | Refl | Tran                   |
   | NTN                      | Symmetric Mode-3-Slice                 |      | TODO                   |
   | NN                       | Independent sum of NN activations(ISN) |      | TODO                   |
   | Plain Bilinear (useless) | symmetric W                            |      | Rank 1, symmetric, PSD |
   | Plain Linear             | v1 equals v2                           |      |                        |
   | Tensor Factorization     | TODO                                   |      | TODO                   |
j. Model the relations as follows:
   # NOTE: Actually remember that what we really want is that
   # If a R_3 b > (a R_1 b)  (a R_2 b) ...
   # Then b R_3 a > (b R_1 a) (b R_2 a)
   # Also note that the entails and rentails relations are impossible to formulate
   # Since they need to be anti-symmetric with entailment.
   # ISN is independent sum of neural network activations : score(a, b) = W(a + b)
   # IDN is independent difference of neural network activations : score(a, b) = W(a - b)
   | SNo | Name      | Approach 1           | Approach 2 | Approach 3                      |
   |   1 | entails   | BilinForm/Impossible | NN/IDN     | NTN/AntiSymmetric Mode-3-slices |
   |   2 | rentails  | BilinForm/Impossible | NN/IDN     | NTN/AntiSymmetric Mode-3-slices |
   |   3 | equiv     | Rank 1, PSD          | NN/ISN     | NTN/Symmetric Mode-3-Slice      |
   |   4 | exclusion | Rank 1, PSD          | NN/ISN     | NTN/Symmetric Mode-3-Slice      |
   |   5 | negation  | Symmetric            | NN/ISN     | NTN/Symmetric Mode-3-Slice      |
   |   6 | cover     | Symmetric            | NN/ISN     | NTN/Symmetric Mode-3-Slice      |
   |   7 | indep     | Rank 1, PSD          | NN/ISN     | NTN/Symmetric Mode-3-Slice      |
k. I would hope to achieve that the numbers in the unprovable case would go up:
   |     | Provable            | UnProvable         |
   | NTN | >= 98.1% (SE 0.67%) | > 87.7% (SE 3.59%) |
   | NN  | >  84.8% (SE 0.84%) | > 68.7% (SE 1.58%) |
l. If the numbers don't go up by introducing these hard constraints. then I can revert to the
   full NN / NTN case and then one by one add constraints of being ISN / IDN to the relations
   For example:
   | Sno | Ap-2(Original) | Ap-2(Constrained) | Ap-2(Entailment) | Ap-2(Others) | Ap-2(Other-1) | Ap-2(Others-[2345]) |
   |   1 | NN             | IDN               | IDN              | NN           | NN            | NN                  |
   |   2 | NN             | IDN               | IDN              | NN           | NN            | NN                  |
   |   3 | NN             | ISN               | NN               | ISN          | ISN           | NN                  |
   |   4 | NN             | ISN               | NN               | ISN          | NN            | ISN                 |
   |   5 | NN             | ISN               | NN               | ISN          | NN            | NN                  |
   |   6 | NN             | ISN               | NN               | ISN          | NN            | NN                  |
   |   7 | NN             | ISN               | NN               | ISN          | NN            | NN                  |
m. In order to make sure that the changes I see are really important changes I would need to do experiments
   on large enough sets, but for now just do it on artificial data, later on use other hierarchies:
   # prf means performance
   |     | Ap-2(Original) | AP-2(Constrained) | Ap-2(Entailment) | Ap-2(Others) | Ap-2(Other-1) | Ap-2(Others-[2345]) |
   | Prf |           87.7 |                97 |               90 |           93 |            88 |                  89 |
n. Bowman extracted 3 types of relations from wordnet, and they actively discard
   equivalent/synonyms by choosing only 1 term out of a cluster of synonyms/equivalents,
   also they remove antonyms and anything else.
   | hypernym     | entails   |
   | hyponym      | rentails  |
   | coordination | exclusion |
   * They limit the size of the vocabulary and extract all of the instances
     of these three relations for single word nouns in WordNet
     that are hyponyms of the node organism.n.01.
   * They balance the distribution of the classes, by slightly downsampling
     instances of the coordinate relation,
   * They had a total of 36,772 relations among 3,217 terms.
   * Embeddings were fixed at 25 dimensions and were initialized randomly or
     using distributional vectors from GloVe.
   * The feature vector produced by the comparison layer was fixed at 80 dimensions.
   * Results were reported using crossvalidation, choosing a disjoint 10% test sample
     for each of five runs.
   Now on the first look their results look incredibly impressive
   Remember this is a 3 class classification problem with 3677 test instances
   and
   |                       <r> |                  |                   |
   |          Portion of Train | NN (Random init) | NTN (Random Init) |
   |---------------------------+------------------+-------------------|
   | (#Train=#Test) (3677) 11% |               91 |                91 |
   |               (11031) 33% |               95 |                95 |
   |              (33094) 100% |               99 |                99 |
   Infact these results are so good that one has to ask what good is it to try and
   improve these results?
o. One can use this to improve the SICK textual entailment challenge?
   For example bowman et. al.'s model reached 76.9% acc. perhaps by adding
   this prior one could improve performance on that task.
   To be honest, I didn't even know about the SICK entailment challenge till bowman
   used it and I don't know if this is a useful task or not? TODO
p. Why am I writing this paper? Really all I want to find out is whether enforcing
   this constraint is a useful guaranteed way to mimic the logical behavior of
   symmetry and transitivity? And also I want to compare different optimization
   techniques for doing so. I am not interested in the SICK challenge. In the end
   I would have a classifier that could predict entailment between two terms. This
   method could be compares with the two vector based distance measures produced
   by mccallum. His method is one of introducing anti-symmetry in the scoring function.
   Once I have those two measures then I'd be able to use these optimization methods to
   learn entailment between phrases. Also I'd be able to credibly talk about sentential
   logic and how to embed it. But not really. I mean this is just the natural logic
   task. The important part is the optimization procedure that incorporates symmetry
   and asymmetry of predicates.
   What other tasks is this type of prior useful for?
   * Well the problem is that I am not learning these constraints of symmetry, This is sort
     just (generic knowledge/a rule) that I am trying to use. Some body tells me that a
     particular relation type is symmetric or asymmetric and I want to learn it.
     The best way to utilize such a knowledge short of changing the representation of
     all the entities in a knowledge base is to just learn a special type of representation
     for the predicate.
   * This is basically learning with a very specific kind of generic knowledge.
   * This could be used in QA over KB or in embedding KB relationships or in updating the embeddings of a KB.
     So some experiments would have to be over freebase / multi-relational data.
     Related work was:
     - Low dimensional embeddings of logic
     - Reasoning with neural networks for KB completion
     - Injecting Logical Background Knowledge into Embeddings for Relation Extraction
     -
q. There are three entailment datasets that I can use
   1. How we blessed distributional semantic evaluation; Baroni
   2. Entailment above the word level in distributional semantics.; Baroni
   3. SICK; Semeval-2014 task 1: Evaluation of compositional distributional
      semantic models on full sentences through semantic relatedness and
      textual entailment. The entailment dataset by Ellie; Baroni.
   But really beyond the idea of phrasal entailment the important idea is of
[1] @proceedings{bowman2014learning,
    Author = {Bowman, Samuel R and Potts, Christopher and Manning, Christopher D},
    Booktitle = {Proceedings of the AAAI Spring Symposium on Knowledge Representation and Reasoning},
    Title = {Learning Distributed Word Representations for Natural Logic Reasoning},
    Year = {2015}}
[2] Bowman Model
    1. Say we have words w1, w2
    2. Find their embeddings e1, e2
       * Optionally transform these embeddings through a tanh layer to get good results.
         Bowman did it for the wordnet data. But it's just crap shoot.
    3. Feed the two into a comparison function. f(e1, e2, Relation)
       Here f(x) is either leaky-relu or tanh
          * f(x) = max(x, 0) + 0.01 min(x, 0).
          * f(x) = tanh(x)
       a. [NN Comparator]  Just uses a v_r.f(W_r[e1, e2] + b_r)
       b. [NTN Comparator] Just uses a v_r.f(e1 . T_r . e2 + W_r[e1, e2] + b_r) or v_r.(f(e1 . T_r . e2) + f(W_r[e1, e2] + b_r))
       c. Bilinear  e1^T W_R e2
       d. Linear    v1 . e1 + v2 . e2
    4. Then feed this into a softmax classifier
    5. Regularize with L2
    6. Train with AdaGrad
'''
