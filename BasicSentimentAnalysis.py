from transformers import AutoTokenizer, AutoModelForSequenceClassification
from newspaper import Article
import torch

# Initialize the tokenizer and model for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Fetch the article
url = 'https://www.aljazeera.com/news/2023/9/16/russia-ukraine-war-list-of-key-events-day-570'
article = Article(url)
article.download()
article.parse()
article.nlp()
text = article.summary

# Tokenize the summary and get the output logits
tokens = tokenizer(text, return_tensors='pt')
output = model(**tokens)[0]

# Compute probabilities and get the label
probs = torch.nn.functional.softmax(output, dim=-1)
label = torch.argmax(probs)

# Translate label to sentiment (0-very negative, 4-very positive)
print(f"Sentiment label: {label.item()}")
