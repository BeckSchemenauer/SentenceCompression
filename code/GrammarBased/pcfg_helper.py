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
        total_count = sum(production_dict[parent].values())
        for child in production_dict[parent]:
            # Normalize each production rule by dividing by the total count
            production_dict[parent][child] = production_dict[parent][child] / total_count

        # Check and adjust for any floating-point precision issues
        final_total = sum(production_dict[parent].values())
        if not (0.99 <= final_total <= 1.01):  # Check if the final total is close to 1
            for child in production_dict[parent]:
                production_dict[parent][child] /= final_total  # Adjust to make the total exactly 1

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
                                    .replace("=1", "-1")  # Replace '=1' with '-X' or remove it
                                    .replace("=2", "-2")  # Replace '=2' with '-X' or remove it
                                    .replace("=3", "-3")  # Replace '=3' with '-X' or remove it
                                    .replace("=4", "-4")  # Replace '=4' with '-X' or remove it
                                    .replace("$", "_DOLLAR_SIGN")
                                    .replace("-LCB-", "LEFT_PAREN")
                                    .replace("-RCB-", "RIGHT_PAREN")
                                    )
                             )
                      )
               )
    )


def fix_nonterminal(nonterminal):
    # Common punctuation replacements
    if nonterminal == ',':
        return 'COMMA'
    elif nonterminal == '.':
        return 'PERIOD'
    elif nonterminal == ':':
        return 'COLON'
    elif nonterminal == '$':
        return 'DOLLAR_SIGN'
    elif nonterminal == '-NONE-':
        return 'NONE'
    elif nonterminal == "``" or nonterminal == "''":
        return "QUOTES"  # Replace the quadruple backticks with a valid name
    elif nonterminal == '-LRB-':
        return 'LEFT_PAREN'
    elif nonterminal == '-RRB-':
        return 'RIGHT_PAREN'

    nonterminal = (nonterminal.replace("''", "'")  # Handle apostrophes
                   .replace("=1", "-1")  # Replace '=1' with '-X' or remove it
                   .replace("=2", "-2")  # Replace '=2' with '-X' or remove it
                   .replace("=3", "-3")  # Replace '=3' with '-X' or remove it
                   .replace("=4", "-4")  # Replace '=4' with '-X' or remove it
                   .replace("$", "_DOLLAR_SIGN")
                   .replace("-LCB-", "LEFT_PAREN")
                   .replace("-RCB-", "RIGHT_PAREN"))

    # Remove everything after "|" if it exists
    nonterminal = re.sub(r"\|.*", "", nonterminal)

    return nonterminal


# Convert production_dict to a string format compatible with nltk.PCFG
def convert_to_pcfg_format(production_dict):
    # Initialize a list to store the formatted rules
    pcfg_rules = []

    # First, accumulate the counts for the modified parents and children
    fixed_production_dict = defaultdict(lambda: defaultdict(float))

    for parent, children_dict in production_dict.items():
        # Fix the parent using the custom function
        fixed_parent = fix_nonterminal(parent)

        for children, count in children_dict.items():
            # Fix each child in the production
            fixed_children = tuple(fix_child(child) for child in children)

            # Add the counts to the new dictionary with fixed labels
            fixed_production_dict[fixed_parent][fixed_children] += count

    # Now, calculate the probabilities based on the fixed productions
    for parent, children_dict in fixed_production_dict.items():
        total_count = sum(children_dict.values())

        # Prevent division by zero (if total_count is 0, skip normalization)
        if total_count == 0:
            print(f"Warning: Total count for {parent} is 0, skipping normalization.")
            continue

        # Format the production rules with updated probabilities
        for children, count in children_dict.items():
            prob = count / total_count  # Recalculate the probability
            children_str = ' '.join(children)  # Join children into a string

            # Format the rule with the recalculated probability
            rule = f"{parent} -> {children_str} [{prob:.8f}]"
            pcfg_rules.append(rule)

    # Return the formatted rules as a string
    return '\n'.join(pcfg_rules)



def get_grammar():
    production_dict = get_production_dict()

    # Generate the PCFG grammar string
    grammar_string = convert_to_pcfg_format(production_dict)
    print(grammar_string)

    # Now you can create the PCFG
    grammar = PCFG.fromstring(grammar_string)

    return grammar