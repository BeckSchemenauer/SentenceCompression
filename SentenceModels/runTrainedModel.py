from transformers import BartTokenizer, BartForConditionalGeneration

# Change this to the directory where the model and other needed files are stored
model_name = "./modelsMSData"
tokenizer2 = BartTokenizer.from_pretrained(model_name)
model2 = BartForConditionalGeneration.from_pretrained(model_name)


def summarize_text_bart(input_text):
    inputs = tokenizer2(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    outputs = model2.generate(**inputs, max_length=50, num_beams=4, early_stopping=True)

    summary = tokenizer2.decode(outputs[0], skip_special_tokens=True)
    return summary


# Example usage
sent1 = "The Java world simply speaking did not tolerate this kind of issue to just go on and on the way that it seems to have here"
sent2 = "For this milestone you and your team should have accomplished the following and report orally to the class or just to me."
sent3 = "In recent years, the importance of mental health awareness has grown significantly, as people around the world begin to understand that mental well-being is just as crucial as physical health, prompting individuals, organizations, and governments alike to invest in resources, education, and policies that support mental health care access and reduce the stigma associated with seeking help."
sent4 = "As the debate around renewable energy intensifies, many countries are faced with the challenge of balancing the economic benefits of fossil fuels with the environmental necessity of adopting cleaner energy sources, a transition that requires not only technological innovation but also a rethinking of global trade, infrastructure, and societal expectations."
sent5 = "Companies are rethinking team dynamics, communication, and work-life balance, but still questioning if traditional office spaces are still needed."
summary = summarize_text_bart(sent4)

print("Original Sentence:")
print(sent4)
summary2 = summarize_text_bart(summary)
summary3 = summarize_text_bart(summary2)
summary4 = summarize_text_bart(summary3)


print("\nSummarized Sentence:")
print(summary)
print(summary2)
print(summary3)
print(summary4)
