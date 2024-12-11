from transformers import BartTokenizer, BartForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt

# Change this to the directory where the model and other needed files are stored
model_name = "../modelsMSData"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def summarize_text_bart(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model.generate(**inputs, max_length=50, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

def compute_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(vectors[0], vectors[1])
    return similarity[0][0]

def compute_compression_ratio(original, summarized):
    return len(summarized) / len(original)

def compute_perplexity_score(text):
    # Simplified readability approximation: lower score implies better readability
    word_count = len(text.split())
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    avg_sentence_length = word_count / max(sentence_count, 1)
    return avg_sentence_length

def analyze_summaries(sentences):
    cosine_sims = []
    compression_ratios = []
    perplexities = []

    for sent in sentences:
        summaries = [sent]
        for _ in range(4):
            summaries.append(summarize_text_bart(summaries[-1]))

        # Compute metrics for each summary level
        cos_sim_levels = [compute_cosine_similarity(summaries[i], summaries[i + 1]) for i in range(len(summaries) - 1)]
        comp_ratio_levels = [compute_compression_ratio(summaries[i], summaries[i + 1]) for i in range(len(summaries) - 1)]
        perplexity_levels = [compute_perplexity_score(summaries[i + 1]) for i in range(len(summaries) - 1)]

        cosine_sims.append(cos_sim_levels)
        compression_ratios.append(comp_ratio_levels)
        perplexities.append(perplexity_levels)

    # Average the scores across all sentences
    avg_cosine_sims = np.mean(cosine_sims, axis=0)
    avg_compression_ratios = np.mean(compression_ratios, axis=0)
    avg_perplexities = np.mean(perplexities, axis=0)

    # Plot results
    x = range(1, 5)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.plot(x, avg_cosine_sims, marker='o')
    plt.title('Average Cosine Similarity')
    plt.xlabel('Summary Level')
    plt.ylabel('Cosine Similarity')

    plt.subplot(1, 3, 2)
    plt.plot(x, avg_compression_ratios, marker='o', color='orange')
    plt.title('Average Compression Ratio')
    plt.xlabel('Summary Level')
    plt.ylabel('Compression Ratio')

    plt.subplot(1, 3, 3)
    plt.plot(x, avg_perplexities, marker='o', color='green')
    plt.title('Average Perplexity Score')
    plt.xlabel('Summary Level')
    plt.ylabel('Perplexity')

    plt.tight_layout()
    plt.show()

# Example usage
sentences = [
    "In recent years, the importance of mental health awareness has grown significantly, as people around the world begin to understand that mental well-being is just as crucial as physical health.",
    "As the debate around renewable energy intensifies, many countries are faced with the challenge of balancing the economic benefits of fossil fuels with the environmental necessity of adopting cleaner energy sources.",
    "Companies are rethinking team dynamics, communication, and work-life balance, but still questioning if traditional office spaces are still needed."
]

analyze_summaries(sentences)