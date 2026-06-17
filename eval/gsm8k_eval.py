import csv
import re
import pandas as pd

def clean_and_extract_number(s):
    # Remove all punctuation and keep only digits
    s = s.replace(',', '')
    s = s.split()[0]
    s = re.sub(r"[^\d]", "", s)

    return int(s) if s else None

def extract_final_number(text):
    """
    Extracts the final numeric value from a string.
    Handles cases like: 
    - "29-1=<<29-1=28>>28 years old."
    - "The answer is: $840$."
    - "Therefore, the final answer is \\boxed{54}."
    """
    if pd.isna(text):
        return None
    text = str(text).split('\n\n')[-1]

    # Priority 1: boxed or LaTeX-style final answer
    match = re.search(r'\\boxed\{(\d+(\.\d+)?)\}', text)
    if match:
        return float(match.group(1))
    
    # Priority 2: $number$ style
    match = re.search(r'\$(\d+(?:\.\d+)?)\$', text)
    if match:
        return float(match.group(1))

    # Priority 3: <<...=number>>number pattern
    match = re.findall(r'>>(\d+(?:\.\d+)?)', text)
    if match:
        return float(match[-1])
    
    # Fallback: get the last number in the string
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    return float(numbers[-1]) if numbers else None

def compute_accuracy(csv_path):
    df = pd.read_csv(csv_path)
    # Clean both columns
    df['GoldClean'] = df['Gold Answer'].apply(extract_final_number)
    df['PredClean'] = df['Prediction'].apply(extract_final_number)
    # print complete prediction, goldclean and predclean columns as a formatted string
    print(df[['GoldClean', 'PredClean']].to_string(index=False))
    # Compute accuracy
    accuracy = (df['GoldClean'] == df['PredClean']).mean()    
    print(f"Accuracy: {accuracy}")

import sys
compute_accuracy(sys.argv[1])
