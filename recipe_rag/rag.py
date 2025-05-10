import pandas as pd
import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import faiss

# -------------------
# 1. Load Data
# -------------------
df = pd.read_csv("recipe_rag/data/full_dataset.csv")

# Parse stringified lists
df['ingredients'] = df['ingredients'].apply(json.loads)
df['directions'] = df['directions'].apply(json.loads)
df['ner'] = df['ner'].apply(json.loads)

df['text'] = df['ingredients'].apply(lambda x: " ".join(x)) + " " + df['directions'].apply(lambda x: " ".join(x))
df['prompt'] = df['ingredients'].apply(lambda x: "Ingredients: " + ", ".join(x))
df['reference'] = df['directions'].apply(lambda x: " ".join(x))

# -------------------
# 2. Embed & Index
# -------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
df['embedding'] = df['text'].apply(lambda x: embedder.encode(x, convert_to_numpy=True))

# FAISS index
embedding_matrix = np.vstack(df['embedding'].values)
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

# -------------------
# 3. Load Generator (e.g., GPT-2)
# -------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# -------------------
# 4. RAG Generation Function
# -------------------
def generate_recipe(prompt, k=5, max_length=200):
    # Embed query
    query_embedding = embedder.encode(prompt, convert_to_numpy=True).reshape(1, -1)

    # Retrieve top-k similar recipes
    _, I = index.search(query_embedding, k=k)
    retrieved = df.iloc[I[0]]['text'].values
    context = " ".join(retrieved)

    # Concatenate prompt + context
    full_input = prompt + "\nContext: " + context
    input_ids = tokenizer.encode(full_input, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=max_length, num_return_sequences=1, do_sample=True)
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

# -------------------
# 5. Evaluation Metrics
# -------------------
def compute_bleu(reference, prediction):
    reference = [reference.split()]
    prediction = prediction.split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu(reference, prediction, smoothing_function=smoothie)

def compute_rouge(reference, prediction):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(reference, prediction)

def compute_meteor(reference, prediction):
    return meteor_score([reference], prediction)

# -------------------
# 6. Evaluation Loop
# -------------------
test_data = df[['prompt', 'reference']].dropna().head(100).to_dict(orient='records')

def evaluate_model(test_data):
    results = {"BLEU": [], "ROUGE-1": [], "ROUGE-2": [], "ROUGE-L": [], "METEOR": []}
    
    for example in test_data:
        prompt = example['prompt']
        reference = example['reference']
        prediction = generate_recipe(prompt)

        results["BLEU"].append(compute_bleu(reference, prediction))
        rouge = compute_rouge(reference, prediction)
        results["ROUGE-1"].append(rouge['rouge1'].fmeasure)
        results["ROUGE-2"].append(rouge['rouge2'].fmeasure)
        results["ROUGE-L"].append(rouge['rougeL'].fmeasure)
        results["METEOR"].append(compute_meteor(reference, prediction))
    
    return {k: round(np.mean(v), 4) for k, v in results.items()}

# -------------------
# 7. Run Evaluation
# -------------------
metrics = evaluate_model(test_data)
print("Evaluation Results:", metrics)
