import pandas as pd
import sys
import subprocess

df = pd.read_csv(sys.argv[1])
df = df.drop('canonical', axis=1)
df = df.fillna('')
df['completion'] = df['completion'].apply(lambda x: x.split('```python')[1].split('```')[0].strip() if isinstance(x, str) and 'python' in x else x)
df.to_json(sys.argv[2], lines=True, orient='records')

subprocess.run([
  "evaluate_functional_correctness",
  sys.argv[2]
])
