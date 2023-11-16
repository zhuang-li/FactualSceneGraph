import functools
import re

import tabulate


def get_seg_list(graphs):
    """
    Extracts segments from a single graph or a list of graphs.

    :param graphs: A single graph string or a list of graph strings.
    :return: A list of unique segments from the graph(s).
    """

    def extract_segments(graph):
        """
        Extract segments from an individual graph string.

        :param graph: A single graph string.
        :return: A list of segments from the graph.
        """
        formatted_graph = space_out_symbols_in_graph(graph)
        return [segment.strip('()').strip() for segment in formatted_graph.split(') , (')]

    if isinstance(graphs, str):
        segments = extract_segments(graphs)
    elif isinstance(graphs, list):
        segments = [seg for graph in graphs for seg in extract_segments(graph)]
    else:
        raise TypeError('Input must be a string or a list of strings')

    # remove the duplicates, note this might influence the evaluation performance
    return list(set(segments))


def space_out_symbols_in_graph(graph_str):
    # Add spaces around parentheses and commas, then split into words
    formatted_str = graph_str.replace('(', ' ( ').replace(')', ' ) ').replace(',', ' , ')

    # Use strip to remove leading/trailing whitespace and join the words back into a string
    return ' '.join(word.strip() for word in formatted_str.split())


def is_graph_format(input_string):
    """
    Check if the input string follows the graph format.

    :param input_string: A string to check.
    :return: True if the string contains elements in parentheses, False otherwise.
    """
    # Pattern to match any content within parentheses
    graph_pattern = r"\(.*?\)"

    return bool(re.search(graph_pattern, input_string))


def clean_graph_string(fact_str):
    # Split the string into individual facts
    facts = fact_str.strip().split(') ,')
    # remove truncated parentheses
    if not fact_str.endswith(')') and len(facts) > 1:
        facts = facts[:-1]
    elif not fact_str.endswith(')') and len(facts) == 1:
        facts = [facts[0].split(',')[0]]
    # Use a set to filter out duplicate facts
    unique_facts = set()
    for fact in facts:
        fact = fact.strip().strip('()').strip()
        if fact:
            unique_facts.add(fact)

    # sort unique_facts

    unique_facts = sorted(unique_facts)

    # Reconstruct the string with unique facts
    unique_fact_str = ' , '.join([f'( {fact} )' for fact in unique_facts])
    return unique_fact_str


def tprint(graph, file=None):
    """
    Print a scene graph as a table.
    The printed strings contain essential information about the parsed scene graph.
    """
    assert isinstance(graph, dict), 'Input must be a dictionary'
    _print = functools.partial(print, file=file)

    _print('Entities:')
    entities_data = [
        [e['head'].lower(), e.get('quantity', ''), ','.join(e.get('attributes', set()))]
        for e in graph['entities']
    ]
    _print(tabulate.tabulate(entities_data, headers=['Entity', 'Quantity', 'Attributes'], tablefmt=_tabulate_format))

    _print('Relations:')
    relations_data = [
        [
            graph['entities'][rel['subject']]['head'].lower(),
            rel['relation'].lower(),
            graph['entities'][rel['object']]['head'].lower()
        ]
        for rel in graph['relations']
    ]
    _print(tabulate.tabulate(relations_data, headers=['Subject', 'Relation', 'Object'], tablefmt=_tabulate_format))


_tabulate_format = tabulate.TableFormat(
    lineabove=tabulate.Line("+", "-", "+", "+"),
    linebelowheader=tabulate.Line("|", "-", "+", "|"),
    linebetweenrows=None,
    linebelow=tabulate.Line("+", "-", "+", "+"),
    headerrow=tabulate.DataRow("|", "|", "|"),
    datarow=tabulate.DataRow("|", "|", "|"),
    padding=1, with_header_hide=None
)



