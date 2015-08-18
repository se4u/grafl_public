BOWDIR = ~/data/vector-entailment/

# Run experiments for investigating the benefits of symmetry and transitivity
# on different types of graphs (chromatic,multi-uni,symmetric/transitive)
# for batch training of edge prediction.
#######################################################################
# TARGET: Learn vector spaced model parameters that can predict edges
# in a dot file. Do cross-validation/progressive validation of those
# scores. This experiment aims to show a single thing, which is that
# if we add symmetry and transitivity constraints to bowman's model
# then we get significant improvement in performance.
# This improvement happens in both the wordnet graphs and other types
# of example entailment graphs.
# OPTIONS:
#   fold : Number of fold
#   test_percentage :
#   train_percentage :
#   architecture : ntn, nn, partsymmetric_ntn, partsymmetric_nn
#   input : A file that contains the entire training data.
#  NUISANCE OPTIONS:
#   Training: Batch Size, Learning Rate calculator, Learning Start Rate,
# Update Calculator, Corpus Shuffler, Dimensionality of representation.
# The actual architecture used, the initialization, the loss function,
# the structure used for learning the metric. The optimizer for these
# hyper-parameters.
# EXPECTATION:
#  I would reproduce these numbers with the flags specified in the
# boxes. And when I change the architecture then the numbers would go up.
#  |            Portion of Train | NN (Random init) | NTN (Random Init) |
#  |-----------------------------+------------------+-------------------+
#  | (#Train=#Test)   (3677) 11% |     (arch=nn) 91 |     (arch=ntn) 91 |
#  |    (train%=30)  (11031) 33% |               95 |                95 |
#  |    (train%=10) (33094) 100% |               99 |                99 |
#########################################################################
b experiment_reproduce_bowman:
	$(MAKE) batch_edgeprediction_architecture:partsymmetric_nn~input:longer_shuffled_synset_relations.tsv

t small_experiment:
	$(MAKE) batch_edgeprediction_architecture:nn~input:toy_synset_relations.tsv
#########################################################################################################
# EXAMPLE:
#  batch_edgeprediction_architecture:partsymmetric_nn~input:longer_shuffled_synset_relations.tsv
#  batch_edgeprediction_architecture:partsymmetric_nn~input:longer_shuffled_synset_relations.tsv
#########################################################################################################
batch_edgeprediction_%:
	./cross_validate --fold 5 --test_percentage 10 --train_percentage 90 $(foreach var,$(subst ~, ,$*),--$(subst :, ,$(var)))

########################################################################
# TARGET: Examples of a library that can learn the model parameters of
#  a vector space model for predicting the edges between two nodes.
#  There are multiple score functions:
#   1. RNTN : Socher/Bowman
#   2. RNN
#   3. Tensor Factorization : Sameer Singh, Tim Rocktaschel
#   4. Gaussian Embeddings : McCallum
#  At present assume that there is only type of edge between two
#   nodes. No multi-relational graphs/rulesets.
#  A
########################################################################


###########################################
# TARGET: Test the graph creation library
###########################################
test_graph_creation_library:



######################################################################
# TARGET: Examples of the capabilities of the graph creation library
# Given a ruleset; R
# Given a skeletal seed graph between the nodes (maybe generated randomly)
# Output either:
#   1. a dot file for the transitive closure
#   2. a dot generated pdf file. with multi-colored edges.
# Of course we can always fake its output. What will we do once we
# have created the `default.output` dot file?
######################################################################
example_graph_creation_library:
	./create_closed_graph --rules default.ruleset --input default.input --stop_after 0 --output default.output



###########################################################################
# CERTIFICATE : Certify that bowman's vocab file and synset_relation file
# have the same information
###########################################################################
certify_bowman_vocab_is_redundant:
	test $(awk '{print $2; print $3}'  $(BOWDIR)/synset-relations/longer_shuffled_synset_relations.tsv  | sort | uniq | wc -l) = $(sort  $(BOWDIR)/synset-relations/longer_wordlist.txt  | uniq | wc -l); exit $?