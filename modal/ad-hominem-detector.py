import modal
import os
import glob
import csv
import json
from pathlib import Path

app = modal.App("ad-hominem-detector")

GPU_CONFIG = "L4"  # or "A10G", etc.

# Chat-gpt estimates
# energy is kwh/
GPU_INFO = {
    "T4": {
        "energy": 0.012,
        "nvidia_driver": "sm_75",
    },
    "L4": {
        "energy": 0.009,
        "nvidia_driver": "sm_89",
    },
    "A10G": {
        "energy": 0.020,
        "nvidia_driver": "sm_86",
    },
    "A100-40GB": {
        "energy": 0.035,
        "nvidia_driver": "sm_80",
    },
    "A100-80GB": {
        "energy": 0.040,
        "nvidia_driver": "sm_80",
    },
    "L40S": {
        "energy": 0.030,
        "nvidia_driver": "sm_89",
    },
    "H100": {
        "energy": 0.050,
        "nvidia_driver": "sm_90",
    },
}
arch = GPU_INFO[GPU_CONFIG]["nvidia_driver"]

model_cache = modal.Volume.from_name("llamacpp-cache", create_if_missing=True)
cache_dir = "/root/.cache/llama.cpp"

results = modal.Volume.from_name("llamacpp-results", create_if_missing=True)
results_dir = "/root/results"

# -------------------- IMAGE -------------------- #


LLAMA_CPP_RELEASE = "b4568"
MINUTES = 60

cuda_version = "12.4.0"  # should be no greater than host CUDA version
flavor = "cudnn-devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}"#-{operating_sys}"


download_image = (
    modal.Image.from_registry(f"nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.12")
    .apt_install(
        "build-essential", "cmake", "git",
        "python3-dev", "python3-pip",
        "libopenblas-dev", "libomp-dev", "clang", "gcc"
    )
    .run_commands([
        "ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH",
        "CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121"
    ])
    # Install other deps
    .pip_install("torch", "pandas", "numpy", "huggingface_hub[hf_transfer]==0.26.2", "transformers", "sentencepiece" )
    
    
    .entrypoint([])
    .env({"LD_LIBRARY_PATH":"/app/:$LD_LIBRARY_PATH"})
    # Add files
    .add_local_file("sample_data_english.csv", remote_path="/root/sample_data_english.csv")
    .add_local_file("sample_data_dutch.csv", remote_path="/root/sample_data_dutch.csv")
    .add_local_file("promt_en.py", remote_path="/root/promt_en.py")
    .add_local_file("promt_nl.py", remote_path="/root/promt_nl.py")
)


from promt_en import prompt as PROMPT_TEMPLATE_EN
from promt_nl import prompt as PROMPT_TEMPLATE_NL


gguf_path = None

# -------------------- DOWNLOAD MODEL -------------------- #
cache_dir = "/root/.cache/llama.cpp"
@app.function(
    image=download_image, volumes={cache_dir: model_cache}, timeout=60 * 10,
)
def download_model(repo_id: str, revision=None, quant: str = "Q4_K_M"):
    global gguf_path
    from huggingface_hub import snapshot_download   
    import shutil

    print("📦 Downloading model from:", repo_id)
    model_path = snapshot_download(repo_id, local_dir=cache_dir)


    gguf_files = glob.glob(os.path.join(model_path, "*.gguf"))
    model_cache.commit()
    print("🦙 model loaded")

    if gguf_files:
        preferred = [f for f in gguf_files if quant in f]
        if not preferred:
            raise FileNotFoundError(f"No GGUF file found for quant '{quant}'")
        
        return preferred[0]
    else:
        raise FileNotFoundError("No GGUF file found in the downloaded model directory.")


# -------------------- RUN LLAMA.CPP -------------------- #

def llama_cpp_inference(llm, gguf_path: str, prompt: str, n_predict: int = -1,DEBUG=False):
    

    # set layers to "off-load to", aka run on, GPU
    if GPU_CONFIG is not None:
        n_gpu_layers = 9999  # all
    else:
        n_gpu_layers = 0

    
    response = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": str(prompt)},
        ],
        temperature=0,
    )
    return response["choices"][0]["message"]["content"]

# -------------------- Helper -------------------- #
def clean_result(text: str):
    import json
    try:
        return json.loads(text)
    except Exception as e:
        print(f"issue during llm-output cleaning: {e}")    
        json_start = text.find('json') + len('json')
        json_end = text.rfind('```')
        json_str = text[json_start:json_end].strip()
        return json.loads(text)
@app.function(
    image=download_image, 
    timeout=60 * 15, 
    volumes={results_dir: results, cache_dir: model_cache}, 
    gpu=GPU_CONFIG)
def gpu_function():
    import subprocess

    import torch

    subprocess.run(["nvidia-smi"])
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())


def get_gpu_power():
    import subprocess
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True
    )
    try:
        return float(result.stdout.strip())
    except:
        return 0.0
power_samples = []

def monitor_power(interval=0.5):
    import time
    while True:
        power = get_gpu_power()
        power_samples.append(power)
        time.sleep(interval)
# -------------------- EVALUATE -------------------- #
cache_dir = "/root/.cache/llama.cpp"
@app.function(
    image=download_image, 
    timeout=60 * 30, 
    volumes={
        results_dir: results,
        cache_dir: model_cache
        },
    gpu=GPU_CONFIG)
def run_pipeline(repo_id, df, prompt, out_path, DEBUG=False):
    import pandas as pd
    import time
    import threading
    from llama_cpp import Llama

    monitor = threading.Thread(target=monitor_power, daemon=True)
    monitor.start()
    start_time = time.time()
    print("🚀 Starting pipeline...")
    gguf_path = download_model.remote(repo_id)

    print("✅ Model downloaded to:", gguf_path)

    # Initialise llm
    llm = Llama(model_path=gguf_path, n_gpu_layers=-1, n_ctx=4096, verbose=DEBUG)
    df = df.head(5)


    predictions = []
    correct = 0
    for idx, row in df.iterrows():
        _prompt = prompt
        true_label = row["Label"].strip()
        speech = row["Speech"].strip()

        print(f"\n🔎 Row {idx}: Label={true_label}")


        full_prompt = _prompt.format(text=speech)
        print("full promt:", full_prompt)
        result = llama_cpp_inference(llm, gguf_path, full_prompt)
        print("🧾 Raw model output:", result)

        # #Calculate tokens
        input_tokens = llm.tokenize(full_prompt.encode("utf-8"))
        input_token_count = len(input_tokens)

        output_tokens = llm.tokenize(result.encode("utf-8"))
        output_token_count = len(output_tokens)

        energy = (input_token_count + output_token_count) / 1000 * GPU_INFO[GPU_CONFIG]["energy"]

        # print(f"Estimated energy used: {energy:.4f} kWh")


        try:
            parsed = clean_result(result)
            predicted_label = "Ad Hominem" if parsed["summary"]["count"] > 0 else "No Ad Hominem"
        except Exception as e:
            print(f"❌ Error parsing result at row {idx}: {e}")
            print("result: ", result)
            predicted_label = "Unknown"
            parsed = result
            

        is_correct = predicted_label == true_label
        correct += int(is_correct)
        print(f"sentence is labled: {true_label} and got detected as {predicted_label} which is correct? {is_correct}")

        predictions.append({
            "Speech": speech,
            "True Label": true_label,
            "Predicted Label": predicted_label,
            "Correct": is_correct,
            "Compute Energy": energy,
            "raw_output": parsed
        })

    accuracy = correct / len(df)
    print(f"🎯 Accuracy: {accuracy:.2%}")

    out_path = os.path.join(results_dir, out_path)
    pd.DataFrame(predictions).to_csv(out_path, index=False)
    print("📁 Results saved to:", out_path)

    duration = time.time() - start_time
    avg_power = sum(power_samples) / len(power_samples)
    energy_measured = avg_power * duration 
    print(f"⚡ Average power during inference (by nvidia-smi): {avg_power:.2f} W")
    print(f"⏱️ Duration: {duration:.2f} s")

    # Assuming predictions is your list of dicts
    df = pd.DataFrame(predictions)
    # Sum the 'Compute Energy' column
    total_energy = df["Compute Energy"].sum()
    energy_token = total_energy / duration

    print(f"🔋 Compute Energy by tokenizer: { energy_token} W")
    return accuracy, energy_measured, energy_token, duration
@app.function(
    image=download_image,
    timeout=60 * 60 * 3,  # 3 hour
    volumes={results_dir: results, cache_dir: model_cache},

)
def run_all_evaluations():
    import pandas as pd
    from promt_en import prompt as PROMPT_TEMPLATE_EN
    from promt_nl import prompt as PROMPT_TEMPLATE_NL

    repo_id = "TheBloke/deepseek-llm-7B-chat-GGUF"
    df = pd.read_csv("sample_data_english.csv")
    df_nl = pd.read_csv("sample_data_dutch.csv")

    accuracy_en_depseek, energy_en_depseek, tokens_en_depseek, duration_en_depseek = run_pipeline.remote(
        repo_id, df, PROMPT_TEMPLATE_EN, "predictions_english_english_deepseek7b.csv"
    )
    accuracy_nl_deepseek, energy_nl_deepseek, tokens_nl_deepseek, duration_nl_deepseek = run_pipeline.remote(
        repo_id, df_nl, PROMPT_TEMPLATE_NL, "predictions_dutch_dutch_deepseek7b.csv"
    )

    repo_id = "BramVanroy/GEITje-7B-ultra-GGUF"
    accuracy_nl_geitje, energy_nl_geitje, tokens_nl_geitje, duration_nl_geitje = run_pipeline.remote(
        repo_id, df_nl, PROMPT_TEMPLATE_NL, "predictions_dutch_dutch_geitje.csv"
    )
    accuracy_en_geitje, energy_en_geitje, tokens_en_geitje, duration_en_geitje = run_pipeline.remote(
        repo_id, df, PROMPT_TEMPLATE_EN, "predictions_english_english_geitje.csv"
    )

    print(f"EN-EN DeepSeek ➤ Accuracy: {accuracy_en_depseek:.2f}, Energy: {energy_en_depseek} J, Tokens: {tokens_en_depseek}, Duration: {duration_en_depseek:.2f}s")
    print(f"NL-NL DeepSeek ➤ Accuracy: {accuracy_nl_deepseek:.2f}, Energy: {energy_nl_deepseek} J, Tokens: {tokens_nl_deepseek}, Duration: {duration_nl_deepseek:.2f}s")
    print(f"EN-EN GEITje     ➤ Accuracy: {accuracy_en_geitje:.2f}, Energy: {energy_en_geitje} J, Tokens: {tokens_en_geitje}, Duration: {duration_en_geitje:.2f}s")
    print(f"NL-NL GEITje     ➤ Accuracy: {accuracy_nl_geitje:.2f}, Energy: {energy_nl_geitje} J, Tokens: {tokens_nl_geitje}, Duration: {duration_nl_geitje:.2f}s")

@app.local_entrypoint()
def main():
    print("📤 Submitting cloud job...")
    run_all_evaluations.remote()
    print("✅ Job submitted! You can now close your laptop.")


