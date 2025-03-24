import modal

# Define the Modal app
app = modal.App("llm-runner")

# Create a serverless image with required dependencies
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
image = modal.Image.debian_slim().pip_install("torch", "transformers")

@app.function(image=image, timeout=300)
def run_llm(prompt: str) -> str:
    """
    Runs a prompt through the LLM and returns the response.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=100)

    return tokenizer.decode(output[0])

# Test function
if __name__ == "__main__":
    prompt = "Tell me about AI."
    print(run_llm.remote(prompt))  # Run function on Modal
