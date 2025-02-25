import csv
import re

import numpy as np
from tqdm import tqdm

epsilons = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
np.random.seed(0)


def get_concpet_name_from_filename(filename: str):
    pattern = r"\[(.*?)\("
    match1 = re.search(pattern, filename.split("/")[-1])
    concept_name = match1.group(1)
    return concept_name


with open(
    "/home/itai/research/PersonalValuesGeometry/experiments/counterfactual_pair.txt"
) as f:
    filenames = [line.strip() for line in f.readlines()]
concept_names = [get_concpet_name_from_filename(filename) for filename in filenames]


for epsilon in epsilons:
    with open(
        f"/home/itai/research/PersonalValuesGeometry/experiments/concept_ranks/eps{epsilon}.csv",
        mode="w",
        newline="",
    ) as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Layer", "Concept Name", "Rank"])

        for n_layer in tqdm(range(1, 28)):
            concept_diff_vectors_path = f"/home/itai/research/PersonalValuesGeometry/matrices/llama-3.2-3b-instruct/layer{n_layer}/valuenet/pos2neg/base/explicit/raw/concept_diff_embeddings.npy"
            concept_diff_vectors = np.load(concept_diff_vectors_path)

            for i, concept_name in enumerate(concept_names):
                diff_vectors = concept_diff_vectors[i].T
                U, S, Vt = np.linalg.svd(diff_vectors)
                rank = np.sum(S > epsilon)
                csv_writer.writerow([n_layer, concept_name, rank])
