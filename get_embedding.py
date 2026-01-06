from datasets import load_dataset, Dataset, load_from_disk 
import pandas as pd
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from datasets import load_dataset_builder, get_dataset_split_names, load_dataset, Dataset
import torch
import torch.nn.functional as F
import pickle
import argparse
from tqdm import tqdm
import time
import sys
import os


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def save_feature(ds, dataset_name, model_name, support_CLS):
    directory = f'benchmark/{dataset_name}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = f'{directory}/embed_{model_name.replace("/", "_")}_{"CLS" if support_CLS else "Avg"}.pkl'

    print(f"=== Saving processed dataset to {filename} ===")
    with open(filename, 'wb') as f:
        pickle.dump({
            "dataset_name": dataset_name,
            # "tags": tags,
            "feature": ds.to_pandas()['features'],
            "nrow": len(ds)
        }, f)


def extract(dataset_name, model_name, batch_size, support_CLS):
    '''
      This function aims to process SPECIFIC dataset
    '''
    print(f"cleaning the raw file of {dataset_name} and saving as HF dataset to benchmark/")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device is {device}")

    # Read raw csv data
    data = pd.read_csv(f"rawdata/{dataset_name}.csv")

    # Convert Pandas DataFrame to Hugging Face Dataset
    ds = Dataset.from_pandas(data)
    print(f"column_names: {ds.column_names}")
    print(f'a total of {len(ds)} rows')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    if model_name == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token
        
    def extract_CLS_features_BERT(batch):
        """Extracts features from a batch of sentences."""
        inputs = tokenizer(batch["sentence"], return_tensors='pt', padding=True, truncation=True)

        # Move tokenized inputs to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # embeddings = outputs[0][:,0,:].numpy()
        embeddings = outputs[0][:, 0, :].cpu().numpy()

        return {'features': embeddings}  # Return features corresponding to [CLS] token

    def extract_embed_features_BERT(batch):
        inputs = tokenizer(batch["sentence"], return_tensors='pt', padding=True, truncation=True)

        # Move tokenized inputs to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Use the model to get the embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Compute the average of all token embeddings
        # avg_embeddings = outputs[0].mean(dim=1).detach().numpy()
        avg_embeddings = outputs[0].mean(dim=1).detach().cpu().numpy()

        return {'features': avg_embeddings}

    def extract_embed_features_GPT2(batch):
        # Tokenize the sentences and convert to input tensors
        inputs = tokenizer(batch['sentence'], return_tensors='pt', padding=True, truncation=True)

        # Move tokenized inputs to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Use the model to get the sequence of hidden-states
        with torch.no_grad():
            outputs = model(**inputs)

        # converting to numpy requires move back to the CPU
        # avg_embeddings = outputs[0].mean(dim=1).numpy()
        avg_embeddings = outputs[0].mean(dim=1).cpu().numpy()

        return {'features': avg_embeddings}

    def extract_embed_features_SBERT(batch):
        # This assumes you have the tokenizer and model defined globally
        inputs = tokenizer(batch["sentence"], return_tensors='pt', padding=True, truncation=True)

        # Move tokenized inputs to GPU
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Use the model to get the embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Perform pooling
        sentence_embeddings = mean_pooling(outputs, inputs['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        # Convert tensor to numpy array
        avg_embeddings = sentence_embeddings.detach().cpu().numpy()

        return {'features': avg_embeddings}

        
    if (model_name == 'bert-base-uncased') and support_CLS:
        ds = ds.map(extract_CLS_features_BERT, batched=True, batch_size=batch_size)
    else:
        if model_name == 'bert-base-uncased':
            method = extract_embed_features_BERT
        elif model_name == 'gpt2':
            method = extract_embed_features_GPT2
        elif model_name == 'sentence-transformers/all-MiniLM-L6-v2':
            method = extract_embed_features_SBERT
        else:
            sys.exit('Unsupported model name')
        ds = ds.map(method, batched=True, batch_size=batch_size)

    save_feature(ds, dataset_name, model_name, support_CLS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one time to generate the extracted features")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="dataset name ('ToxicCommentChallenge', 'hate_speech_offensive')")
    parser.add_argument("--model_name", type=str, required=True, help="embedding model ('gpt2', 'bert', "
                                                                      "'sentence-transformers/all-MiniLM-L6-v2')")
    parser.add_argument("--cls", type=bool, required=False, default=False, help="Use CLS as sentence embedding or not")
    parser.add_argument("--batch_size", type=int, required=False, default=320)
    args = parser.parse_args()

    extract(args.dataset_name, args.model_name, args.batch_size, args.cls)
