from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
SEMANTIC_MODEL_ID = "all-MiniLM-L6-v2"

def main():
    print(f"Downloading: {MODEL_ID}")
    AutoTokenizer.from_pretrained(MODEL_ID)
    AutoModelForCausalLM.from_pretrained(MODEL_ID)
    print(f"Downloading: {SEMANTIC_MODEL_ID}")
    SentenceTransformer(SEMANTIC_MODEL_ID)
    print("Done.")

if __name__ == "__main__":
    main()