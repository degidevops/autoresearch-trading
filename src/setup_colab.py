# src/setup_colab.py
import os
import subprocess
from pathlib import Path

def setup_colab():
    print("--- Setting up environment for Google Colab ---")
    
    # 1. Update requirements
    req_file = Path("requirements.txt")
    with open(req_file, "a") as f:
        f.write("\nllama-cpp-python")
    
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
    
    # 2. Download a base GGUF model if requested
    if os.getenv("DOWNLOAD_MODEL") == "true":
        print("Downloading base model...")
        os.system("wget -q https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF/resolve/main/llama-3-8b-instruct.Q4_K_M.gguf -O /content/model.gguf")
        os.environ["LOCAL_MODEL_PATH"] = "/content/model.gguf"
        print("Model downloaded to /content/model.gguf")

if __name__ == "__main__":
    setup_colab()
