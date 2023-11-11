import functools
import tabulate

def space_out_symbols_in_graph(graph_str):
    # Add spaces around parentheses and commas, then split into words
    formatted_str = graph_str.replace('(', ' ( ').replace(')', ' ) ').replace(',', ' , ')

    # Use strip to remove leading/trailing whitespace and join the words back into a string
    return ' '.join(word.strip() for word in formatted_str.split())


def remove_redundant_facts(fact_str):
    # Split the string into individual facts
    facts = fact_str.strip().split(') ,')

    # Use a set to filter out duplicate facts
    unique_facts = set()
    for fact in facts:
        fact = fact.strip().strip('()').strip()
        if fact:
            unique_facts.add(fact)

    # Reconstruct the string with unique facts
    unique_fact_str = ' , '.join([f'( {fact} )' for fact in unique_facts])
    return unique_fact_str


def tprint(graph, file=None):
    """
    Print a scene graph as a table.
    The printed strings contain essential information about the parsed scene graph.
    """

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
