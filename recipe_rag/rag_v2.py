import pandas as pd
import json
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# ---------------------
# CONFIG
# ---------------------
CSV_PATH = "data/full_dataset.csv"
CHUNK_SIZE = 1000
MAX_ROWS = 10000
EMBED_MODEL = "all-MiniLM-L6-v2"

# ---------------------
# STEP 1: Load + Preprocess in Chunks
# ---------------------
embedder = SentenceTransformer(EMBED_MODEL)
data = []
rows_loaded = 0

for chunk in pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE):
    for _, row in chunk.iterrows():
        if rows_loaded >= MAX_ROWS:
            break
        try:
            ingredients = json.loads(row['ingredients'])
            directions = json.loads(row['directions'])
            prompt = "Ingredients: " + ", ".join(ingredients)
            reference = " ".join(directions)
            text = " ".join(ingredients) + " " + reference
            embedding = embedder.encode(text)

            data.append({
                "prompt": prompt,
                "reference": reference,
                "embedding": embedding
            })
            rows_loaded += 1
        except Exception:
            continue
    if rows_loaded >= MAX_ROWS:
        break

df = pd.DataFrame(data)

# ---------------------
# STEP 2: Build FAISS Index
# ---------------------
embedding_matrix = np.vstack(df['embedding'].values)
embedding_dim = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embedding_matrix)

# ---------------------
# STEP 3: Load Generator
# ---------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# ---------------------
# STEP 4: RAG Generation
# ---------------------
def generate_recipe(prompt, k=5, max_length=200):
    query_embedding = embedder.encode(prompt, convert_to_numpy=True).reshape(1, -1)
    _, I = index.search(query_embedding, k=k)
    retrieved_texts = df.iloc[I[0]]['prompt'].values
    context = "\n".join(retrieved_texts)
    input_text = prompt + "\nContext:\n" + context
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=max_length, do_sample=True, top_k=50, top_p=0.95)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ---------------------
# STEP 5: Evaluation Metrics
# ---------------------
def compute_bleu(reference, prediction):
    reference = [reference.split()]
    prediction = prediction.split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu(reference, prediction, smoothing_function=smoothie)

def compute_rouge(reference, prediction):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }

def compute_meteor(reference, prediction):
    return meteor_score([reference], prediction)

# ---------------------
# STEP 6: Evaluation with DataFrame Return
# ---------------------
def evaluate_model(df, sample_size=10):
    eval_logs = []

    for i in tqdm(range(min(sample_size, len(df)))):
        prompt = df.iloc[i]['prompt']
        reference = df.iloc[i]['reference']
        prediction = generate_recipe(prompt)

        bleu = compute_bleu(reference, prediction)
        rouge = compute_rouge(reference, prediction)
        meteor = compute_meteor(reference, prediction)

        eval_logs.append({
            "prompt": prompt,
            "reference": reference,
            "prediction": prediction,
            "BLEU": bleu,
            "ROUGE-1": rouge['rouge1'],
            "ROUGE-2": rouge['rouge2'],
            "ROUGE-L": rouge['rougeL'],
            "METEOR": meteor
        })

    eval_df = pd.DataFrame(eval_logs)
    return eval_df

# ---------------------
# STEP 7: Run + Print Evaluation Summary
# ---------------------
eval_df = evaluate_model(df, sample_size=10)
print("\n\n💡 Evaluation Metrics Summary:")
print(eval_df[["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "METEOR"]].mean().round(4))

# Save results (optional)
eval_df.to_csv("recipe_rag_eval_results.csv", index=False)
