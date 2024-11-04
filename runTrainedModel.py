from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BartTokenizer, BartForConditionalGeneration

model_name = "H:/models2"
tokenizer2 = BartTokenizer.from_pretrained(model_name)
model2 = BartForConditionalGeneration.from_pretrained(model_name)

def summarize_text_bart(input_text):
    # Tokenize the input text
    inputs = tokenizer2(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Generate summary using the model
    outputs = model2.generate(**inputs, max_length=50, num_beams=4, early_stopping=True)

    # Decode the generated output
    summary = tokenizer2.decode(outputs[0], skip_special_tokens=True)
    return summary

# Example usage
new_sentence = "The Java world simply speaking did not tolerate this kind of issue to just go on and on the way that it seems to have here"
summary = summarize_text_bart(new_sentence)

print("Original Sentence:")
print(new_sentence)
print("\nSummarized Sentence:")
print(summary)