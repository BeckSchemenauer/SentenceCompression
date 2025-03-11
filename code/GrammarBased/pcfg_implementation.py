import pickle
from pcfg_helper import get_grammar
import nltk
from itertools import combinations
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from nltk import word_tokenize, TreebankWordDetokenizer
import language_tool_python


class BertGrammarChecker:
    def __init__(self, model_name="textattack/bert-base-uncased-CoLA"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.model.eval()  # Set the model to evaluation mode

    def check_grammar(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()
        return predicted_label == 1


# Initialize tools for grammaticality checking
tool = language_tool_python.LanguageTool('en-US')
checker = BertGrammarChecker()


def load_or_generate_grammar(grammar_file="grammar.pkl"):
    try:
        with open(grammar_file, "rb") as file:
            grammar = pickle.load(file)
            print("Grammar loaded from file.")
    except (FileNotFoundError, EOFError):
        print("Grammar file not found. Generating grammar...")
        grammar = get_grammar()
        with open(grammar_file, "wb") as file:
            pickle.dump(grammar, file)
            print("Grammar saved to file.")
    return grammar


def find_probability(sequence, grammar):
    def get_rule_probability(parent, children):
        for prod in grammar.productions(lhs=nltk.Nonterminal(parent)):
            if prod.rhs() == tuple(children):
                return prod.prob()
        return 0.0

    total_probability = 1.0
    start_index = 0
    while start_index < len(sequence):
        matched = False
        for end_index in range(len(sequence), start_index, -1):
            subsequence = sequence[start_index:end_index]
            for parent in grammar._lhs_index.keys():
                probability = get_rule_probability(parent.symbol(), subsequence)
                if probability > 0:
                    total_probability *= probability
                    start_index = end_index
                    matched = True
                    break
            if matched:
                break
        if not matched:
            return 0.0
    return total_probability


def get_token_combinations(tokens):
    result = []
    for r in range(1, len(tokens) + 1):
        result.extend([list(combo) for combo in combinations(tokens, r)])
    return result


def rejoin_tokens(token_combinations):
    detokenizer = TreebankWordDetokenizer()
    return [detokenizer.detokenize(combo) for combo in token_combinations]


# Main code
grammar = load_or_generate_grammar()
tokens = word_tokenize("The quick brown fox jumps over the lazy dog.")
all_combos = get_token_combinations(tokens)

# List to store constructions and their probabilities
constructions_with_probs = []

# Evaluate each token combination
total_combos = len(all_combos)
for i, combo in enumerate(all_combos, 1):
    print(f"Processing combination {i}/{total_combos}...", end="\r")
    pos_tags = nltk.pos_tag(combo)
    pos_sequence = [tag for _, tag in pos_tags]

    probability = find_probability(pos_sequence, grammar)
    if probability > 0:
        rejoined_sentence = " ".join(combo)

        # Check grammaticality using both tools
        matches = len(tool.check(rejoined_sentence)) == 0
        is_grammatical_by_checker = checker.check_grammar(rejoined_sentence)

        if matches and is_grammatical_by_checker:
            constructions_with_probs.append((rejoined_sentence, probability))

print("\nProcessing complete. Sorting results...")

# Sort the constructions by probability in descending order and get the top 5
top_5_constructions = sorted(constructions_with_probs, key=lambda x: x[1], reverse=True)[:5]

# Output the results
print("Top 5 Most Probable Grammatical Constructions:")
for sentence, prob in top_5_constructions:
    print(f"Sentence: '{sentence}' with Probability: {prob:.8f}")
