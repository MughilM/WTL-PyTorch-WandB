# WTL-PyTorch
This repository supports the paper by me, Samrat Halder, Yuval Marton, and Asad Sayeed, titled 
*Where's the Learning in Representation Learning for Compositional Semantics and the Case of Thematic Fit*
([arXiv](https://arxiv.org/abs/2208.04749)).
The [original repository](https://github.com/MughilM/RW-Eng-v3-src) contains the original training and evaluation code.
This process uses TensorFlow as the main framework, and also manually keeps track of the various hyperparameters,
such as embedding size, type of model, etc.

Since then, many new frameworks have been released and have matured. This repository will use the same methodology,
but will use PyTorch, PyTorch Lightning, and Weights and Biases as the main frameworks for training, and model
metric tracking. This should reduce the amount of boilerplate code that is necessary to maintain.

Finally, at the time of writing, the original repository does not contain the code to pull and process the original
data. Suffice to say, this repository will indeed include that.

Please see below for the main dependencies this repository has been tested for. For full list, please see 
corresponding environment file in the `envs` folder.

- PyTorch 2
- PyTorch Lightning 2
- CUDA 12.1

Licensing is identical to the original repository. For details about the dataset used, please reference the 
original repository.

For the CSV file detailing the original corpus file names and its file IDs, the `files.list` Google Drive API
function was used along with my own API key. Please see below for the code snippet used to generate `data/corpus_files.csv`.

```python
import requests
import pandas as pd

# Define the API key and URL
API_KEY = '<API_KEY>'
URL = 'https://www.googleapis.com/drive/v3/files'
params = {'key': API_KEY, 'q': "'1me3S-Gxiup8-64fT3X5rGgE5P_esyhRm' in parents",
          'pageToken': ''}
# The initial request. The result will have a 'nextPageToken' key which we use to
# get the additional results.
result = requests.get(url=URL, params=params).json()
# Create empty DataFrame
df = pd.DataFrame(columns=['id', 'name', 'url'])

# We keep going until there is no more nextPageToken
while 'nextPageToken' in result.keys():
    # Grab the variables we need and add it to the df we have
    sub_df = pd.DataFrame(result['files']).drop(columns=['kind', 'mimeType'])
    sub_df['url'] = 'https://drive.google.com/uc?id=' + sub_df['id']
    df = pd.concat((df, sub_df))
    # Grab the next page of results...
    params['pageToken'] = result['nextPageToken']
    result = requests.get(url=URL, params=params).json()

# Get the last page of results...
sub_df = pd.DataFrame(result['files']).drop(columns=['kind', 'mimeType'])
sub_df['url'] = 'https://drive.google.com/uc?id=' + sub_df['id']
df = pd.concat((df, sub_df))

# Next, extract the actual ID number of each gz file, and put in a separate
# column, and then sort it in ascending order.
df['num'] = df['name'].str.extract(r'heads.ukwac.fixed-nowiki.node(\d+).converted.xml.converted.xml.gz').astype(int)
df.sort_values('num', ascending=True, inplace=True)

df.to_csv('./corpus_files.csv', index=False)  # ...Moved to data folder after the fact
```

