import torch
import pandas as pd
import re
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler

# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import pyarrow as pa
import pyarrow.dataset as ds
from datasets import Dataset
from transformers import get_scheduler
from tqdm.auto import tqdm

with open('stock_data.csv', encoding='utf8') as csvfile:
    df = pd.read_csv(csvfile, delimiter=',')

df.dropna(axis=0, how='any', inplace=True)                         # Excludes null-containing rows
num_positive = df['Sentiment'].value_counts()[1]

df.loc[df['Sentiment'] == -1, 'Sentiment'] = 0
num_negative = df['Sentiment'].value_counts()[0]
# print(df['Sentiment'].value_counts())

# Hyperparameters
word_frequency_requirement = 0.0013*(df['Sentiment'].size) # the number of times a word has to appear to be given
# it's own encoding. All words under this limit are encoded as the same 'unknown' word.
sg = 0
vector_size = 1000

# Regex removal of various undesirable parts of a tweet
def clean_tweet(tweet):
  tweet = re.sub(r"@[A-Za-z0-9]+", ' ', tweet) # Twitter handle removal
  tweet = re.sub(r"https?://[A-Za-z0-9./]+", ' ', tweet) # URL removal
  tweet = re.sub(r"[']", "", tweet) # Apostrophe removal
  tweet = re.sub(r"[^a-zA-Z.!?]", ' ', tweet) # Remove symbols that are not alphabetic or sentence endings
  tweet = re.sub(r"([^a-zA-Z])", r" \1 ", tweet) # Places spaces around sentence endings,
  # so they are encoded as their own words, rather than being lumped in with other words.
  tweet = re.sub(r" +", ' ', tweet) # Excess whitespace removal
  tweet = tweet.lower() # Send tweet to lowercase
  return tweet

# Prepare word lemmatizer and stopwords list for sanitisation
lemmatizer = WordNetLemmatizer()
stops = set(stopwords.words('english'))

def sanitise(tweet):
    tweet = clean_tweet(tweet)
    tweet = filter(lambda w: w not in stops, tweet.strip().split()) # Remove stopwords
    return " ".join(list(map(lemmatizer.lemmatize, tweet))) # Lemmatize words.



# Hyperparameters
train_proportion = 0.80
hidden_layer_size = 70
learning_rate = 0.0001
batch_size = 64
epochs = 10000

# print(df['Text'].tolist())
# tweets = df['Text'].map(sanitise).tolist() 
# labels = df['Sentiment'].tolist()
print(type(df))
df['Text'] = df['Text'].map(sanitise)
max_tweet_length = max(len(x) for x in df['Text'])
print(max_tweet_length)
train_df = df.sample(frac = 0.8)
test_df = df.drop(train_df.index)


dataset = ds.dataset(pa.Table.from_pandas(train_df).to_batches())

### convert to Huggingface dataset
# hg_dataset = Dataset(pa.Table.from_pandas(df))
train_dataset = Dataset(pa.Table.from_pandas(train_df))
# test_dataset
# df = df.to_dict()
# print(df)
# print(hg_dataset[0])



tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["Text"], padding="max_length", truncation=True)


tokenized_datasets = train_dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(["Text"])
tokenized_datasets = tokenized_datasets.remove_columns(["__index_level_0__"])
tokenized_datasets = tokenized_datasets.rename_column("Sentiment", "labels")
tokenized_datasets.set_format("torch")
print(tokenized_datasets)
# class twitterDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels
        
#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         return item
#     def __len__(self):
#         return len(self.labels)
    
# twitter_dataset = twitterDataset(inputs, labels)

# print(len(dataset_dataloader)) # 5791


train_dataloader = DataLoader(dataset = tokenized_datasets, batch_size = batch_size, shuffle = True)
# test_dataloader = torch.utils.data.DataLoader(dataset = test_tensor, batch_size = 1)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(torch.cuda.is_available())
model.to(device)


progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        
metric = load_metric("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()