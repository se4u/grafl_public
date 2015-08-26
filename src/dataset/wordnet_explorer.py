'''
| Filename    : wordnet_explorer.py
| Description : A few functions for exploring wordnet.
| Author      : Pushpendre Rastogi
| Created     : Fri Aug  7 14:42:05 2015 (-0400)
| Last-Updated:
|           By:
|     Update #: 21
'''
import sys


def inadmissible(name):
    return (not name.endswith('n.01')) or '_' in name


def all_hyponyms_of_synset(synset):
    ''' Find all the hyponym entities of this synset in wordnet.
    Returns a set of wordnet synsets.
    Params
    ------
    synset : A wordnet synset object
    '''
    def all_hyponyms_impl(synset, hyponym_list):
        hyp_list = synset.hyponyms()
        for hyponym_synset in hyp_list:
            name = hyponym_synset.name()
            if not inadmissible(name):
                hyponym_list.append(hyponym_synset)
                all_hyponyms_impl(hyponym_synset, hyponym_list)
        return

    l = [synset]
    all_hyponyms_impl(synset, l)
    return set(l)


def all_hyponym_edges_in_tree(root):
    ''' Find all the hyponym direct edges inside the tree rooted at
    root. Returns a set of edges going from parent(hypernym) to child(hyponym).

    Params
    ------
    root : A wordnet synset object
    '''
    def all_hyponym_edges_in_tree_impl(root):
        root_name = root.name()
        hyp_list = root.hyponyms()
        l = []
        for hyp in hyp_list:
            hyp_name = hyp.name()
            if not inadmissible(hyp_name):
                l.append((root_name, hyp_name))
                l.extend(all_hyponym_edges_in_tree_impl(hyp))
        return l
    return set(all_hyponym_edges_in_tree_impl(root))


def synset_name(synset):
    return synset.split('.')[0]

if __name__ == '__main__':
    from nltk.corpus import wordnet as wn
    root = wn.synset('organism.n.01')
    ah = all_hyponyms_of_synset(root)
    ahe = all_hyponym_edges_in_tree(root)
    print >> sys.stderr, len(ah)
    print >> sys.stderr, len(ahe)

    for (a, b) in ahe:
        print synset_name(a), synset_name(b)
