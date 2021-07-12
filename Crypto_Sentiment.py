#!/usr/bin/env python
# coding: utf-8

# In[13]:


# Initial imports
import os
import pandas as pd
from dotenv import load_dotenv
import nltk as nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


api_key = os.getenv("NEWS_API_KEY")
type(api_key)


# In[15]:


# Create a newsapi client
from newsapi import NewsApiClient
newsapi = NewsApiClient(api_key=api_key)
print(newsapi)


# In[16]:


# Fetch the Bitcoin news articles
Bitcoin_headlines = newsapi.get_everything(
    q="Bitcoin",
    language="en",
    
)
# Print total articles
print(f"Total articles about Bitcoin: {Bitcoin_headlines['totalResults']}")

# Show sample article
Bitcoin_headlines["articles"][0]


# In[17]:


# Fetch the Ethereum news articles
Ethereum_headlines = newsapi.get_everything(
    q="Ethereum",
    language="en",
    
)
# Print total articles
print(f"Total articles about Ethereum: {Ethereum_headlines['totalResults']}")

# Show sample article
Ethereum_headlines["articles"][0]


# In[35]:


# Create the Bitcoin sentiment scores DataFrame
Bitcoin_sentiments = []

for article in Bitcoin_headlines["articles"]:
    try:
        text = article["content"]
        
        sentiment = analyzer.polarity_scores(text)
        compound = sentiment["compound"]
        pos = sentiment["pos"]
        neu = sentiment["neu"]
        neg = sentiment["neg"]
        
        Bitcoin_sentiments.append({
            
            
           "Text": text,
            "Date": date,
            "Compound": compound,
            "Positive": pos,
            "Negative": neg,
            "Neutral": neu
        })
        
    except AttributeError:
        pass
    
# Create DataFrame
Bitcoin_df = pd.DataFrame(Bitcoin_sentiments)

# Reorder DataFrame columns
cols = ["Date", "Text", "Compound", "Positive", "Negative", "Neutral"]
Bitcoin_df = Bitcoin_df[cols]

Bitcoin_df.head()


# In[30]:


# Create the ethereum sentiment scores DataFrame
Ethereum_sentiments = []

for article in Ethereum_headlines["articles"]:
    try:
        text = article["content"]
        
        sentiment = analyzer.polarity_scores(text)
        compound = sentiment["compound"]
        pos = sentiment["pos"]
        neu = sentiment["neu"]
        neg = sentiment["neg"]
        
        Ethereum_sentiments.append({
            
            
           "Text": text,
            "Date": date,
            "Compound": compound,
            "Positive": pos,
            "Negative": neg,
            "Neutral": neu
        })
        
    except AttributeError:
        pass
    
# Create DataFrame
Ethereum_df = pd.DataFrame(Ethereum_sentiments)

# Reorder DataFrame columns
cols = ["Date", "Text", "Compound", "Positive", "Negative", "Neutral"]
Ethereum_df = Ethereum_df[cols]

Ethereum_df.head()


# In[36]:


# Describe the Bitcoin Sentiment
Bitcoin_df.describe()


# In[37]:


# Describe the Ethereum Sentiment
Ethereum_df.describe()


# ### Questions
# Q: Which coin had the highest mean positive score?
# 
# A: Bitcoin.
# 
# Q: Which coin had the highest compound score?
# 
# A: Ethereum 
# 
# Q. Which coin had the highest positive score?
# 
# A: Ethereum.

# ### Tokenizer
# 
# In this section, you will use NLTK and Python to tokenize the text for each coin. Be sure to:
# 
# 1.Lowercase each word
# 
# 2.Remove Punctuation
# 
# 3.Remove Stopwords
# 
# 

# In[39]:


from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from string import punctuation
import re
import nltk
nltk.download('stopwords')


# In[40]:


lemmatizer = WordNetLemmatizer()


# In[41]:


# Complete the tokenizer function- stopwords, punctuations, lower-case
def tokenizer(text):
    sw= set(stopwords.words('english'))
    regex= re.compile("[^a-zA-Z ]")
    
    re_clean = regex.sub('', str(text))
    words= word_tokenize(re_clean)
    lem=[lemmatizer.lemmatize(word) for word in words]
    tokens= [word.lower() for word in lem if word.lower() not in sw ]
    
    return tokens


# In[42]:


# Create a new tokens column for bitcoin
Bitcoin_df["tokens"] = Bitcoin_df.Text.apply(tokenizer)
Bitcoin_df.head()


# In[43]:


# Create a new tokens column for ethereum
Ethereum_df["tokens"] = Ethereum_df.Text.apply(tokenizer)
Ethereum_df.head()


# ### NGrams and Frequency Analysis¶
# 
# In this section you will look at the ngrams and word frequency for each coin.
# 1.Use NLTK to produce the n-grams for N = 2
# 
# 2.List the top 10 words for each coin.
# 
# 

# In[44]:


from collections import Counter
from nltk import ngrams


# In[45]:


# Tokenized Bitcoin articles
Bitcoin_p = tokenizer(Bitcoin_df.Text.str.cat())
Bitcoin_p


# In[46]:


# Tokenized Ethereum articles
Ethereum_p= tokenizer(Ethereum_df.Text.str.cat())
Ethereum_p


# In[50]:


# Generate the Bitcoin N-grams where N=2
N=2
Bigram_counts_Bit = Counter(ngrams(Bitcoin_p, N))
print(dict(Bigram_counts_Bit))


# In[53]:


# Generate the Ethereum N-grams where N=2
N= 2
Bigram_counts_Eth = Counter(ngrams(Ethereum_p, N))
print(dict(Bigram_counts_Eth))


# ### Side Mission to show most common works for each of the Bigrams

# In[56]:


Bigram_counts_Bit.most_common(10)


# In[54]:


Bigram_counts_Eth.most_common(10)


# ### Using actual "token_count" to find most common word from each coin.
# 
# #### Will look up why there's a difference

# In[55]:


# Use the token_count function to generate the top 10 words from each coin
def token_count(tokens, N=10):
    """Returns the top N tokens from the frequency count"""
    return Counter(tokens).most_common(N)


# In[58]:


# Get the top 10 words for Bitcoin
bitcoin_common= token_count(Bitcoin_p, 10)
bitcoin_common


# In[59]:


# Get the top 10 words for Ethereum
ethereum_common= token_count(Ethereum_p,10)
ethereum_common


# In[60]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [20.0, 10.0]


# In[62]:


# Generate the Bitcoin word cloud
wordcloud = WordCloud(colormap="RdYlBu").generate(Bitcoin_df.Text.str.cat())
plt.imshow(wordcloud)
plt.axis("off")
fontdict = {"fontsize": 50, "fontweight": "bold"}
plt.title("Bitcoin Word Cloud", fontdict=fontdict)
plt.show()

## In experimenting the words shift around, El Salvador was in the bottom left and now in mid-right

### Also El Salvador is in the Cloud do to them deciding Bitcoin is legal tender.


# In[63]:


# Generate the Ethereum word cloud
wordcloud = WordCloud(colormap="RdYlBu").generate(Ethereum_df.Text.str.cat())
plt.imshow(wordcloud)
plt.axis("off")
fontdict = {"fontsize": 50, "fontweight": "bold"}
plt.title("Ethereum Word Cloud", fontdict=fontdict)
plt.show()

## Nothing too exciting in the articles for Ethereum, there is NFT which is the current sorta ponzi scheme thing going.


# ### Named Entity Recognition
# 
# In this section, you will build a named entity recognition model for both coins and visualize the tags using SpaCy.
# 

# In[64]:


import spacy
from spacy import displacy


# In[65]:


# Optional - download a language model for SpaCy
get_ipython().system('python -m spacy download en_core_web_sm')


# In[66]:


# Load the spaCy model
nlp = spacy.load('en_core_web_sm')


# ### Bitcoin NER

# In[72]:


# Concatenate all of the bitcoin text together
all_concat_Bitcoin = Bitcoin_df.Text.str.cat()
all_concat_Bitcoin


# In[73]:


# Run the NER processor on all of the text
Btc_doc = nlp(all_concat_Bitcoin)
Btc_doc
# Add a title to the document
Btc_doc.user_data["title"] = "Bitcoin NER"


# In[74]:


# Render the visualization
colors = {"ORG": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
options = {"ents": ["ORG"], "colors": colors}
displacy.render(Btc_doc,style='ent')


# In[71]:


# List all Entities
for ent in Btc_doc.ents:
    print(ent.text, ent.label_)


# ### Ethereum NER

# In[75]:


# Concatenate all of the ethereum text together
all_concat_Ethereum = Ethereum_df.Text.str.cat()
all_concat_Ethereum


# In[76]:


# Run the NER processor on all of the text
Eth_doc = nlp(all_concat_Ethereum)
Eth_doc
# Add a title to the document
Eth_doc.user_data["title"] = "Ethereum NER"


# In[77]:


# Render the visualization
colors = {"ORG": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
options = {"ents": ["ORG"], "colors": colors}
displacy.render(Eth_doc,style='ent')


# In[78]:


# List all Entities
for ent in Eth_doc.ents:
    print(ent.text, ent.label_)


# In[ ]:




