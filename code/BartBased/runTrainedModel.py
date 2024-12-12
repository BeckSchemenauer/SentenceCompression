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

        # Compute metrics for each summary level compared to the original sentence
        cos_sim_levels = [compute_cosine_similarity(sent, summaries[i + 1]) for i in range(len(summaries) - 1)]
        comp_ratio_levels = [compute_compression_ratio(sent, summaries[i + 1]) for i in range(len(summaries) - 1)]
        perplexity_levels = [compute_perplexity_score(summaries[i + 1]) for i in range(len(summaries) - 1)]

        cosine_sims.append(cos_sim_levels)
        compression_ratios.append(comp_ratio_levels)
        perplexities.append(perplexity_levels)

    # Average the scores across all sentences
    avg_cosine_sims = np.mean(cosine_sims, axis=0)
    avg_compression_ratios = np.mean(compression_ratios, axis=0)
    avg_perplexities = np.mean(perplexities, axis=0)

    # Plot results
    x = [1, 2, 3, 4]  # Corrected order for summary levels
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
    "In an era where artificial intelligence and machine learning are rapidly advancing, organizations are grappling with the challenge of integrating these technologies into existing infrastructures while ensuring ethical considerations are not overlooked.",
    "As climate change continues to pose a significant threat to our planet, governments and industries are increasingly adopting renewable energy solutions to mitigate environmental damage and promote sustainable development.",
    "The recent breakthroughs in gene editing technologies, such as CRISPR, have opened up unprecedented opportunities for scientists to address genetic disorders and improve agricultural yields, but they also raise complex ethical and regulatory concerns.",
    "In the digital age, data privacy has become a paramount concern, with individuals, corporations, and governments striving to balance the benefits of data sharing with the need to protect sensitive information.",
    "The proliferation of social media platforms has transformed the way people communicate, share information, and form relationships, yet it has also introduced new challenges related to misinformation, cyberbullying, and mental health.",
    "Advances in quantum computing have the potential to revolutionize industries ranging from cryptography to pharmaceutical development, but the technology still faces significant technical and practical hurdles before widespread adoption can occur.",
    "The rise of e-commerce and the increasing demand for fast, reliable delivery services have reshaped the logistics industry, driving innovations in supply chain management, automation, and last-mile delivery solutions.",
    "In modern cities, the concept of smart infrastructure is gaining traction, with technologies such as IoT sensors and AI-driven analytics enhancing urban planning, traffic management, and resource optimization.",
    "As space exploration enters a new era with the involvement of private companies, there is growing excitement about the potential for lunar bases, Mars colonization, and the commercial exploitation of asteroid resources.",
    "In recent years, telemedicine has emerged as a transformative approach to healthcare delivery, enabling patients to access medical consultations and treatments remotely, especially in underserved and rural areas.",
    "The global pandemic has underscored the importance of scientific collaboration and innovation, leading to the rapid development and deployment of vaccines to combat COVID-19 and its variants.",
    "As autonomous vehicles move closer to becoming a reality, the automotive industry faces challenges related to regulatory compliance, public safety, and the ethical programming of AI decision-making systems.",
    "The debate around the ethical use of facial recognition technology has intensified, with critics raising concerns about privacy violations, racial bias, and the potential misuse of surveillance systems.",
    "With the rapid growth of the gig economy, workers are demanding better protections, benefits, and fair wages, prompting policymakers to reconsider traditional labor laws and employment frameworks.",
    "The integration of augmented reality and virtual reality technologies into education has created immersive learning experiences that engage students and enhance their understanding of complex subjects.",
    "The increasing reliance on cloud computing services has raised questions about data sovereignty, security, and the environmental impact of large-scale data centers.",
    "In the field of biotechnology, synthetic biology is enabling researchers to engineer organisms for applications ranging from biofuel production to the creation of new medical therapies.",
    "The expansion of high-speed internet access in remote and underserved regions is helping to bridge the digital divide, fostering economic growth, and improving access to education and healthcare.",
    "As consumer preferences shift toward sustainable and ethically sourced products, companies are rethinking their supply chains to align with environmental, social, and governance (ESG) goals.",
    "The transition to a circular economy model is encouraging industries to focus on reducing waste, reusing resources, and designing products for a longer lifecycle, promoting environmental sustainability."
]


analyze_summaries(sentences)
