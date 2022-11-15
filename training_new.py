from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
from datasets import Dataset
import torch
import pyarrow as pa
import pandas as pd
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

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
batch_size = 4

print(type(df))
df['Text'] = df['Text'].map(sanitise)
max_tweet_length = max(len(x) for x in df['Text'])
print(max_tweet_length)
train_df = df.sample(frac = 0.8)
eval_df = df.drop(train_df.index)

### convert to Huggingface dataset
train_dataset = Dataset(pa.Table.from_pandas(train_df))
eval_dataset = Dataset(pa.Table.from_pandas(eval_df))


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["Text"], padding="max_length", truncation=True)


tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True)
tokenized_train_datasets = tokenized_train_datasets.remove_columns(["Text", "__index_level_0__"])
tokenized_train_datasets = tokenized_train_datasets.rename_column("Sentiment", "labels")
tokenized_train_datasets.set_format("torch")

tokenized_eval_datasets = eval_dataset.map(tokenize_function, batched=True)
tokenized_eval_datasets = tokenized_eval_datasets.remove_columns(["Text", "__index_level_0__"])
tokenized_eval_datasets = tokenized_eval_datasets.rename_column("Sentiment", "labels")
tokenized_eval_datasets.set_format("torch")
print(tokenized_eval_datasets)

train_dataloader = DataLoader(dataset = tokenized_train_datasets, batch_size = batch_size, shuffle = True)
eval_dataloader = DataLoader(dataset = tokenized_eval_datasets, batch_size = batch_size, shuffle = True)

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
#device = torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(torch.cuda.is_available())
model.to(device)


optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 10
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)




plot_loss = []
plot_correct = []

model.train()
for epoch in range(num_epochs):
    batch_progress_bar = tqdm(total=len(train_dataloader))
    epoch_loss = 0.
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        epoch_loss += loss.data.item()
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        batch_progress_bar.update(1)
        batch_progress_bar.set_description("Progress through epoch")

    batch_progress_bar.clear()

    eval_batch_progress_bar = tqdm(total=len(eval_dataloader))
    correct = 0
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            output = model(**batch)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        correct += torch.sum(predictions==batch["labels"]).data.item()
        eval_batch_progress_bar.update(1)
        eval_batch_progress_bar.set_description("Epoch Accuracy evaluation progress")

    eval_batch_progress_bar.clear()

    accuracy = correct/(num_negative+num_positive)*100
    print("Epoch: %02d, Loss: %f, Accuracy: %.2f%%" % (epoch+1, epoch_loss, accuracy) )

    plot_loss.append(loss/len(train_dataloader))
    plot_correct.append(accuracy)
    
        
# Plot results
plt.plot(plot_loss)
plt.xlabel('Epoch')
plt.ylabel('Avg. Loss per Epoch (on Training Set)')
plt.show()

plt.plot(plot_correct)
plt.xlabel('Epoch')
plt.ylabel('Accuracy per Epoch (on Test Set)')
plt.show()