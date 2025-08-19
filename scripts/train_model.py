from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_from_disk, Dataset 
import pandas as pd 

#Load preprocessed data 
train_df = pd.read.csv('../data/processed/train.csv')
test_df = pd.read.csv('../data/processed/test.csv')
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.form_pandas(test_df)

#Tokenizer (Simpified from Notebook)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize(batch):
    return tokenizer(examples['Email Text'], padding = 'max_length', truncation=true, max_length=512)

tokenizer_train = train_dataset.map(tokenize, batched=True)
tokenizer_test = test_dataset.map(tokenize, batched=True)
tokenizer_train = tokenized_train.rename_column('label', 'labels')
tokenizer_test  = tokenized_test.rename_column('label', 'labels')

#Training Arguments 
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
trainiga_args = TrainingArguments(Output_dir='../results', num_train_epochs=3, per_device_train_batch_size=8)
trainer = Trainer(model=model, args=training_args, train_dataset=tokenizer_train, eval_dataset=tokenizer_test) 
trainer.train()

#Save the model 
model.save_pretrained('../models/bert_model')
tokenizer.save_pretrained('../models/bert-model')
