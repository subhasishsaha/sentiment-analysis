from transformers import AutoModel, AutoTokenizer
import os

model_name = "FacebookAI/roberta-base"

print(f"Downloading and saving model for '{model_name}'...")

# Load the model and tokenizer from the Hub
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save them to your local directory
model.save_pretrained("Models/roberta_base")
tokenizer.save_pretrained("Models/roberta_tokenizer")

print(f"Successfully saved. You can now run your Streamlit app.")