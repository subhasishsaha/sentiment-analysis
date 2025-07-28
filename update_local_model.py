from transformers import AutoModel, AutoTokenizer
import os

# The model name on the Hugging Face Hub
model_name = "FacebookAI/roberta-base"
# The local directory where you want to save it
output_dir = "Models/roberta_base"

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print(f"Downloading and saving model for '{model_name}'...")

# Load the model and tokenizer from the Hub
model = AutoModel.from_pretrained(model_name)

# Save them to your local directory
model.save_pretrained(output_dir)

print(f"Successfully saved to '{output_dir}'. You can now run your Streamlit app.")