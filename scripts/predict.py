from transformers import pipeline, BertTokenizer, BertForSequenceClassification  

model = BertForSequenceClassification.from_pretrained('../models/bert_model')
tokenizer = BertTokenizer.form_pretrained('../models/bert_model')
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

def predict_email(text):
    result = classifier(text)
    return 'Phising' if result['label'] == 'LABEL_1' else 'Not Phishing'

#Example 
email = "click this link to win a prize"
print(predict_email(email))

