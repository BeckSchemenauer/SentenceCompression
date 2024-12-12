from itertools import combinations
import spacy

from matplotlib import pyplot as plt
from matplotlib_venn import venn2
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from nltk import word_tokenize, TreebankWordDetokenizer
import language_tool_python


class BertGrammarChecker:
    def __init__(self, model_name="textattack/bert-base-uncased-CoLA"):
        """
        Initialize the grammar checker with the specified BERT model.
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.model.eval()  # Set the model to evaluation mode

    def check_grammar(self, sentence):
        """
        Check if a sentence represented by tokens is grammatically correct.

        Args:
        - tokens (list of str): The input sentence as a list of tokens.

        Returns:
        - bool: True if the sentence is grammatically correct, False otherwise.
        """

        # Tokenize the input for the BERT model
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

        # Forward pass through the model
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get predicted label
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1).item()

        # Return True for correct grammar (LABEL_1), False otherwise
        return predicted_label == 1


def get_token_combinations(tokens):
    # Generate all combinations of tokens
    result = []
    for r in range(1, len(tokens) + 1):
        result.extend([list(combo) for combo in combinations(tokens, r)])
    return result


def rejoin_tokens(token_combinations):
    detokenizer = TreebankWordDetokenizer()
    return [detokenizer.detokenize(combo) for combo in token_combinations]


# Initialize tools
tool = language_tool_python.LanguageTool('en-US')
checker = BertGrammarChecker()

# Example sentence
tokens = word_tokenize("The quick brown fox jumps over the lazy dog.")
all_combos = get_token_combinations(tokens)
rejoined_sentences = rejoin_tokens(all_combos)

# Collect results
results = []

for i, sentence in enumerate(rejoined_sentences):
    print(f"{i + 1}/{len(rejoined_sentences)}")

    # Check grammaticality using both methods
    matches = len(tool.check(sentence)) == 0  # True if tool says correct
    is_grammatical_by_checker = checker.check_grammar(sentence)  # True if checker says correct

    # Store results
    results.append((sentence, matches, is_grammatical_by_checker))

    if matches and is_grammatical_by_checker:
        print(sentence)

# Categorize results
tool_correct = {r[0] for r in results if r[1]}  # Sentences tool said are correct
checker_correct = {r[0] for r in results if r[2]}  # Sentences checker said are correct

tool_incorrect = {r[0] for r in results if not r[1]}  # Sentences tool said are incorrect
checker_incorrect = {r[0] for r in results if not r[2]}  # Sentences checker said are incorrect

# Create Venn diagrams
plt.figure(figsize=(12, 6))

# Correct sentences
plt.subplot(1, 2, 1)
venn2(
    [tool_correct, checker_correct],
    set_labels=('Python Language Tool Correct', 'BERT Checker Correct')
)
plt.title('Intersection of Correct Sentences')

# Incorrect sentences
plt.subplot(1, 2, 2)
venn2(
    [tool_incorrect, checker_incorrect],
    set_labels=('Python Language Tool Incorrect', 'BERT Checker Incorrect')
)
plt.title('Intersection of Incorrect Sentences')

plt.tight_layout()
plt.show()

# Print examples
print("Examples of Correct Sentences by Both:")
print("\n".join(list(tool_correct & checker_correct)[:5]))

print("\nExamples of Incorrect Sentences by Both:")
print("\n".join(list(tool_incorrect & checker_incorrect)[:5]))

# sent = word_tokenize("The quick brown fox jumps over the lazy dog.")
#
# combos = get_token_combinations(sent)
#
# for combo in combos:
#     is_grammatical = checker.check_grammar(combo)
#     if is_grammatical:
#         print(combo)
