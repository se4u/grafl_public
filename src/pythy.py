'''
| Filename    : wordnet_explorer.py
| Description : A few functions for exploring wordnet.
| Author      : Pushpendre Rastogi
| Created     : Fri Aug  7 14:42:05 2015 (-0400)
| Last-Updated:
|           By:
|     Update #: 10
'''


def all_hyponyms_impl(synset, hyponym_list):
    # Prune out the entire sub-tree if its root is not a
    # single word, sense 1 noun.
    inadmissible = lambda name: not name.endswith('n.01') or '_' in name

    hyp_list = synset.hyponyms()
    for hyponym_synset in hyp_list:
        name = hyponym_synset.name()
        if not inadmissible(name):
            hyponym_list.append(hyponym_synset)
            all_hyponyms_impl(
                hyponym_synset, hyponym_list)
    return


def all_hyponyms(synset):
    l = [synset]
    all_hyponyms_impl(synset, l)
    return l

if __name__ == '__main__':
    from nltk.corpus import wordnet as wn
    print len(all_hyponyms(wn.synset('organism.n.01')))
    print len(set(all_hyponyms(wn.synset('organism.n.01'))))
