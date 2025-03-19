import random

from ..prompts import PROMPT_TEMPLATES, schwartz_values_instructions

random.seed(42)


# PROMPT_TEMPLATES = {
#     "reflection": "Consider the deeper meaning behind this: {}\nThis text reflects values related to:",
#     "analysis": "Analyzing the underlying principles in: {}\nThe core values expressed here are:",
#     "implicit": "Looking at the implicit meaning of: {}\nThe fundamental values shown are:",
#     "explicit": "What values are represented in this text: {}\nThe text demonstrates values of:",
#     "bare": "{}",
#     "theme": "What is the main theme in this text: {}\nKey themes include:",
#     "topic": "What is the main topic in this text: {}\nMain topic covered are:",
# }


def get_sequence_pairs(filepath: str, num_samples: int = 1000) -> list[list[str]]:
    with open(filepath, encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) > num_samples:
        lines = random.sample(lines, num_samples)
    pairs = [line.split("\t") for line in lines]

    return pairs


def get_counterfactual_pairs(
    counter_facutual_file_path: str, prompt_type: str, num_sample: int = 1000
) -> tuple[list[str], list[str]]:
    """
    counterfactual pairが格納されたtxtファイルを読み込み、prompt_typeに従ってpromptを生成する
    Returns:
        positive_sequences: list[str]
        negative_sequences: list[str]
    """
    if prompt_type not in PROMPT_TEMPLATES:
        raise ValueError(
            f"Invalid prompt_type: {prompt_type}. Available types: {list(PROMPT_TEMPLATES.keys())}"
        )
    prompt_template = PROMPT_TEMPLATES[prompt_type]

    # counterfactual pairが格納されたtxtファイルを読み込む
    with open(counter_facutual_file_path) as f:
        lines = f.readlines()

    if len(lines) > num_sample:
        lines = random.sample(lines, num_sample)

    text_pairs = []
    for line in lines:
        if line.strip():
            base, target = line.strip().split("\t")
            if "schwartz" in prompt_type:
                prefixed_base = prompt_template.format(schwartz_values_instructions, base)
                prefixed_target = prompt_template.format(schwartz_values_instructions, target)
            else:
                prefixed_base = prompt_template.format(base)
                prefixed_target = (prompt_template.format(target))
            text_pairs.append((prefixed_base, prefixed_target))

    positive_sequences = []
    negative_sequences = []

    for base, target in text_pairs:
        positive_sequences.append(base)
        negative_sequences.append(target)

    return positive_sequences, negative_sequences
