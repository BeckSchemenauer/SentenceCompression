import nltk
from nltk import PCFG
from nltk.corpus import treebank
from collections import defaultdict
import re


def get_production_dict():
    # Initialize dictionary to store parent -> {child sequence: count}
    production_dict = defaultdict(lambda: defaultdict(float))

    def process_tree(tree):
        # Check if the current node is a Tree (non-leaf node)
        if isinstance(tree, nltk.Tree):
            # Get the label of the parent node
            parent = tree.label()

            # Check for productions with terminal children (words)
            if len(tree) == 1 and isinstance(tree[0], str):  # Terminal case
                # Production rule for a terminal
                word = tree[0]
                production_dict[parent][(word,)] += 1
            else:
                # Non-terminal case
                children = tuple(child.label() if isinstance(child, nltk.Tree) else child for child in tree)
                production_dict[parent][children] += 1

            # Recursively process each child node
            for child in tree:
                process_tree(child)

    # Process each parsed sentence in the Brown corpus (or use treebank for this example)
    for tree in treebank.parsed_sents():
        process_tree(tree)

    # Normalize to get probabilities
    for parent in production_dict:
        count = sum(production_dict[parent].values())
        for child in production_dict[parent]:
            production_dict[parent][child] = production_dict[parent][child] / count

    return production_dict


def fix_child(child):
    # Handle special case like AT&T
    if child == "AT&T":
        return f'"{child}"'

    # Handle known terminals that are uppercase but not part of rules
    if child.isupper() and re.match(r'[A-Z]+[&]?[A-Z]+', child):
        return f'"{child}"'  # Treat as terminal by enclosing in quotes

    # Replace non-standard symbols and handle terminal symbols
    return f'"{child}"' if not child.isupper() else (
        re.sub(r'\.([A-Z])\.', r'\1',  # Handle '.A.' format
               re.sub(r'([A-Z])\.', r'\1',  # Handle 'A.' format
                      re.sub(r"^'([A-Za-z])", r"\1",  # Handle "'S" to "S" format
                             re.sub(r"\*([A-Z]+)\*", r"\1",  # Handle '*T*' to 'T'
                                    child.replace("-NONE-", "NONE")  # Handle '-NONE-' replacement
                                    .replace("-LRB-", "\"(\"")  # Replace left round bracket
                                    .replace("-RRB-", "\")\"")  # Replace right round bracket
                                    .replace("''", "'")  # Handle apostrophes
                                    .replace("=1", "-X")  # Replace '=1' with '-X' or remove it
                                    .replace("=2", "-X")  # Replace '=2' with '-X' or remove it
                                    .replace("=3", "-X")  # Replace '=3' with '-X' or remove it
                                    .replace("=4", "-X")  # Replace '=4' with '-X' or remove it
                                    .replace("$", "")
                                    )
                             )
                      )
               )
    )


def fix_nonterminal(nonterminal):
    if nonterminal == ',':
        return 'COMMA'
    elif nonterminal == '.':
        return 'PERIOD'
    elif nonterminal == '-NONE-':
        return 'NONE'
    # Add other punctuation cases as needed
    return nonterminal


# Convert production_dict to a string format compatible with nltk.PCFG
def convert_to_pcfg_format(production_dict):
    pcfg_rules = []
    for parent, children_dict in production_dict.items():

        fixed_parent = fix_nonterminal(parent)

        for children, prob in children_dict.items():
            # Process each child:
            children_str = ' '.join(fix_child(child) for child in children)

            # Format each rule with the probability
            rule = f"{fixed_parent} -> {children_str} [{prob:.8f}]"
            pcfg_rules.append(rule)

    return '\n'.join(pcfg_rules)


def get_grammar(production_dict):
    # Generate the PCFG grammar string
    grammar_string = convert_to_pcfg_format(production_dict)
    print(grammar_string)

    # Now you can create the PCFG
    grammar = PCFG.fromstring(grammar_string)

    return grammar


get_grammar(get_production_dict())
