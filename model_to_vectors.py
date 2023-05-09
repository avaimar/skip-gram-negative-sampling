"""
Converts the in embeddings to the appropriate format for evaluation and saves them in
a directory accessible to the ConditionalEmbeddings repo. Need to run performance.py from
the ConditionalEmbeddings repo to evaluate.
"""

import argparse
from model import SkipGramEmbeddings
from pathlib import Path
from argparse import Namespace
from gensim.corpora import Dictionary
import numpy as np
import torch
import tqdm
from tqdm.contrib.concurrent import process_map


def load_model(model_path: str, vocab_path: str) -> SkipGramEmbeddings:
    torch_model = torch.load(
        model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # noinspection PyTypeChecker
    # NOTE: vocab is type int:str
    vocab = Dictionary().load(str(vocab_path))

    # Load model
    model = SkipGramEmbeddings(len(vocab),
                               #torch_model["args"].embedding_len
    300)

    model.load_state_dict(torch_model["state_dict"])
    model.vocab = vocab
    model.word_input_embeddings = {}
    for word_i, vec in zip(model.vocab.keys(), model.input_embeddings()):
        word = model.vocab[word_i]
        model.word_input_embeddings[word] = vec

    return model


def get_embedding(model: SkipGramEmbeddings, word: str):
    return torch.tensor(model.word_input_embeddings[word]).tolist()


def compute_decade_embeddings(
    model: SkipGramEmbeddings, output_embedding_path: str
):
    all_words = list(model.vocab.values())
    embeddings = []
    for word in tqdm.tqdm(all_words, desc="Word", position=2):
        embeddings.append(get_embedding(model, word))

    # Write out in w2v format
    with open(output_embedding_path, "w") as f:
        for word, embedding in zip(all_words, embeddings):
            f.write(f"{word} {' '.join(map(str, embedding))}\n")


def main(args):
    torch.set_grad_enabled(False)
    model = load_model(
        args.source_dir / 'results' / f"model_best_SGNSCOHA_{args.run_id}.pth",
        args.source_dir / 'data' / f'dictionary_{args.decade_int}0s.pth',
    )
    compute_decade_embeddings(
        model, args.save_dir / f"decade_embeddings_{args.file_stamp}_{args.run_id}_{args.decade_int}.txt"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-run_id", type=str, required=True)
    parser.add_argument("-name", type=str, required=True)
    parser.add_argument("-run_location", type=str, choices=['local', 'sherlock'])
    parser.add_argument("-decade_int", type=int, required=True)
    parser.add_argument("-source_dir", type=str, required=False)
    parser.add_argument("-save_dir", type=str, required=False)
    parser.add_argument("-file_stamp", type=str, required=False)

    args = parser.parse_args()

    if args.decade_int > 1000:
        raise Exception('Decade should be 3 integers.')

    if args.run_location == 'sherlock':
        args.source_dir = Path(f'/oak/stanford/groups/deho/legal_nlp/WEB/data/{args.name}/SGNS-repo')
        args.save_dir = Path(f'/oak/stanford/groups/deho/legal_nlp/WEB/data/{args.name}/results')
    elif args.run_location == 'local':
        args.source_dir = Path(__file__).parent
        args.save_dir = Path(__file__).parent / '..' / 'ConditionalEmbeddings' / 'data' / args.name / 'results'
    args.file_stamp = args.name
    main(args)
