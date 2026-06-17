import csv

def index_to_option(idx):
    return f"({chr(ord('A') + int(idx))})"

import re

def find_and_format_A_to_D(text):
    # Pattern matches:
    # 1. (A) style
    # 2. $$\boxed{A}$$ style
    # 3. $$\boxed{\text{A}}$$ style
    pattern = re.compile(
        r'\(([A-D])\)'                               # group 1: (A)
        r'|\$\$\\boxed\{([A-D])\}\$\$'               # group 2: $$\boxed{A}$$
        r'|\$\$\\boxed\{\\text\{([A-D])\}\}\$\$',    # group 3: $$\boxed{\text{A}}$$
        re.MULTILINE
    )

    formatted_matches = []
    for match in re.findall(pattern, text):
        # match is a tuple like ('B', '', '') or ('', 'C', '') etc.
        letter = next(filter(None, match))  # pick the non-empty group
        formatted_matches.append(f"({letter})")

    return formatted_matches[-1] if len(formatted_matches) > 0 else None

def extract_option_from_response(response, gt=None):
    if gt is not None:
        gt = gt.strip()
        if response.find(gt) != -1:
            return gt
        else:
            return find_and_format_A_to_D(response.replace('\n', '').strip())
    return None  # fallback in case of invalid response

def calculate_accuracy(csv_file_path):
    total = 0
    correct = 0

    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            if len(row) < 3:
                continue  # skip malformed rows
            gt = row[-3].strip()
            predicted_response = row[-2].strip()
            correct_option = gt[:4].strip() # extract_option_from_response(gt)
            predicted_option = extract_option_from_response(predicted_response, correct_option)

            total += 1
            print("-------------------- ")
            print(predicted_option, correct_option)
            print("-------------------- ")
            if predicted_option == correct_option:
                correct += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"Accuracy: {accuracy * 100:.2f}%")

import sys

calculate_accuracy(sys.argv[1])
