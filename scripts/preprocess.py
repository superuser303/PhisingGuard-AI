import pandas as pd
from datasets import Load_dataset 
from sklearn.model_selection import train_test_split 
import nlkt 
from nlkt.tokenize import word_tokenize 
nlkt.download('punkt')

def preprocess_data( save_path='../data/processed/')
    dataset = Load_dataset("zenfang-liu/phising-email-dataset")
    df = dataset['train'].to_pandas()
    df['label'] = df['Email Type'].map({'Safe Email': 0, 'Phising Email': 1})
    df=df [['Email Text', 'Label']].dropna()

    def clean_text(text):
        tokens = word_tokenize(text.lower())              
        return ' '.joint(tokens)
    
    df['Email Text'] = df['Email Text'].apply(clean_text)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Label'])
    train_df.to_csv(sw)