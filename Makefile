BOWDIR = ~/data/vector-entailment/
FN_TO_PARAM = $(foreach var,$(subst ~, ,$*),--$(subst :, ,$(var)))

all: res/bowman_wordnet_longer_shuffled_synset_relations.tsv
	echo $(basename $<)
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

# src/train.py --yaml res/experiments/crossvalidate.yaml
t small_experiment:
	$(MAKE) batch_edgeprediction_architecture:nn~input:hypernymOf_partOf.default.input.tsv~fold:1

#########################################################################
# EXAMPLE: See experiment_reproduce_bowman and small_experiment
# --fold 5 --test_percentage 10 --train_percentage 90
#########################################################################
batch_edgeprediction_%:
	python src/cross_validate.py $(FN_TO_PARAM)

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
	python src/util_make_toy_graph.py --rules default.ruleset --input default_input.dot --stop_after 0 --output default_output.dot

example_convert_tsv_to_map: res/bowman_wordnet_longer_shuffled_synset_relations.tsv
	awk '{print $$1}' $< | sort | uniq > $(basename $<).map; \
	awk '{print $$2; print $$3}' $< | sort | uniq | nl -v 0  >> $(basename $<).map

example_convert_dot_to_tsv:
	python src/util_make_toy_graph.py --output res/hypernymOf_partOf.default.input.tsv


###########################################################################
# CERTIFICATE : Certify that bowman's vocab file and synset_relation file
# have the same information
###########################################################################
certify_bowman_vocab_is_redundant:
	test $(awk '{print $2; print $3}'  $(BOWDIR)/synset-relations/longer_shuffled_synset_relations.tsv  | sort | uniq | wc -l) = $(sort  $(BOWDIR)/synset-relations/longer_wordlist.txt  | uniq | wc -l); exit $?
