'''
| Filename    : util_make_toy_graph.py
| Description : Make toy graph datasets with specified nodes, edges and composition rules by performing a clozure.
| Author      : Pushpendre Rastogi
| Created     : Tue Aug 18 17:56:59 2015 (-0400)
| Last-Updated: Tue Aug 18 19:22:41 2015 (-0400)
|           By: Pushpendre Rastogi
|     Update #: 9
'''
import pydot
import config
def main(args):
    if args.output.endswith('tsv'):
        if not args.perform_closure:
            graph = pydot.graph_from_dot_file(args.input)
            edges = graph.get_edges()
            with open(args.output, "wb") as f:
                for edge in edges:
                    src = edge.get_source()
                    dest = edge.get_destination()
                    edge_type = edge.get('type') # edge.type
                    f.write('%s %s %s\n'%(src, dest, edge_type))
            return

    raise NotImplementedError

if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser(description='Util Make Toy Graphs')
    arg_parser.add_argument('--seed', default=0, type=int, help='Default={0}')
    arg_parser.add_argument('--rules', default='hypernymOf_partOf.ruleset', type=str, help='Default={default.ruleset}')
    arg_parser.add_argument('--input', default='hypernymOf_partOf.default.input.dot', type=str, help='Default={default_input.dot}')
    arg_parser.add_argument('--stop_after', default=-1, type=int, help='Default={-1}')
    arg_parser.add_argument('--perform_closure', default=0, type=int, help='Default={1}')
    arg_parser.add_argument('--output', default='default_output.tsv', type=str, help='Default={default_output.dot}')
    args=arg_parser.parse_args()
    # import random
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    main(args)
