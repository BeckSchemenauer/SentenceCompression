import spacy
from spacy import displacy
from nltk import sent_tokenize

# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")


def preprocess(sentence):
    # Process the sentence
    doc = nlp(sentence)
    return doc


def compress(doc):
    sentence = []

    for token in doc:
        # Get part-of-speech and other attributes
        pos = token.pos_

        # Check various conditions to exclude words
        if (
            pos in ["ADV", "ADJ"]           # Exclude adverbs and adjectives
            or token.is_stop                # Exclude stop words
            or token.is_punct               # Exclude punctuation
            or len(token) < 3               # Exclude short words
            or token.lemma_ in ["be", "have", "do"]  # Exclude specific verbs
        ):
            continue

        sentence.append(token)

    recombined_sentence = "".join([token.text_with_ws for token in sentence])

    return recombined_sentence


def process_paragraph(paragraph):
    # Process the paragraph as a single document
    doc = nlp(paragraph)
    compressed_sentences = []

    # Iterate over each sentence in the paragraph
    for sent in doc.sents:
        compressed_sentence = compress(sent)
        compressed_sentences.append((sent, compressed_sentence))

    return compressed_sentences


def process_sentence(input_sentence):
    doc = preprocess(input_sentence)
    compressed_sentence = compress(doc)
    return compressed_sentence


doc_ = preprocess("The quick brown fox jumped over the lazy dog.")

# # Print POS tags and dependencies
for token in doc_:
    print(f"{token.text:<12} POS: {token.pos_:<6} Dependency: {token.dep_:<10} Head: {token.head.text}")

# Render the dependency tree and save it as an SVG file
svg = displacy.render(doc_, style="dep", jupyter=False)
with open("dependency_tree.svg", "w", encoding="utf-8") as file:
    file.write(svg)

print("Dependency tree saved as 'dependency_tree.svg'")

print(process_sentence("The quick brown fox jumped over the lazy dog."))

# with open("test1.txt", 'r') as file:
#     text = file.read()
#
# cs = process_paragraph(text)
# for s, c in cs:
#     print(s)
#     print(c)
#     print()
