import spacy
from itertools import combinations
from pcfg import get_pcfg

# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")
pcfg = get_pcfg()


def token_combinations(tokens):
    result = []
    for r in range(1, len(tokens) + 1):
        result.extend([list(combo) for combo in combinations(tokens, r)])
    return result


def preprocess(sentence):
    # Process the sentence
    doc = nlp(sentence)
    return doc


def calculate_pcfg_score(sequence, pcfg, parse_rules):
    """
    Calculate the PCFG score for a given sequence of tokens.

    Parameters:
    - sequence: List of tokens in the sentence (e.g., ["the", "dog", "walks"])
    - pcfg: Dictionary with production rules as keys and their probabilities as values.
    - parse_rules: List of production rules used in parsing the sequence.

    Returns:
    - score: The PCFG score for the sequence.
    """
    score = 1.0

    for rule in parse_rules:
        if rule in pcfg:
            score *= pcfg[rule]
        else:
            # If a rule is not in the PCFG, assume probability 0 or some low value
            score *= 0  # This would invalidate the sequence
            break

    return score


sentence = "The quick brown fox jumps over the lazy dog"

doc = preprocess(sentence)

sequences = token_combinations(doc)

for sequence in sequences:
    calculate_pcfg_score(sequences, pcfg, parse_rules=[])