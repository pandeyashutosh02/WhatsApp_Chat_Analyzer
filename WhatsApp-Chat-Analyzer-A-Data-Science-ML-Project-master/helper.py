from urlextract import URLExtract
from wordcloud import WordCloud,ImageColorGenerator
from collections import Counter
import pandas as pd
import emoji
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from PIL import Image
import streamlit as st
extactor = URLExtract()

def stats(data,user):

    if(user != "Overall"):
        data =  data[data['User'] == user]

    total_msg = data.shape[0]  # total msgs

    words = []     # total words
    for x in data["message"]:
            words.extend(x.split())
    val = "<Media omitted>\n"
    if val  in data["message"]:
        total_media = data['message'].value_counts()["<Media omitted>\n"]
    else:
        total_media =0

    link  = []
    for msg in data["message"]:
        link.extend(extactor.find_urls(msg))  # total link

    return total_msg, len(words),total_media , len(link)



def busy_user(data):
     data = data[data["User"] != "Group Notification"] 
     val = data["User"].value_counts().head()
     percetange_of_each_user =  round( (data["User"].value_counts()/data.shape[0])*100 , 2)
     return val,percetange_of_each_user


def create_word_cloud(data , user):
    if(user != "Overall"):
        data =  data[data['User'] == user]
    temp_data = data[data["User"] != "Group Notification"]  # remove grp notification
    temp_data = temp_data[temp_data["message"]!= "<Media omitted>\n"]  # remove media_omttied
    
    f = open("stopwords.txt" ,'r')
    stop_words = f.read()
    stop_words = stop_words.split()

    def remove_stop_words(msg):
        y = []
        for x in msg.lower().split():
            if x not in stop_words:
                 y.append(x)
        return " ".join(y)
    
    temp_data["message"] = temp_data["message"].apply(remove_stop_words)
   
    mask = np.array(Image.open('wp.jpg'))
    mask[mask == 0] = 255
    wc = WordCloud(width = 500 , height = 500 , min_font_size =10 ,background_color = 'black',mask =mask)
    df_wc = wc.generate(temp_data["message"].str.cat(sep = " "))
    return df_wc


def most_common_word(data,user):
    if(user != "Overall"):
        data =  data[data['User'] == user]
    temp_data = data[data["User"] != "Group Notification"]  # remove grp notification
    temp_data = temp_data[temp_data["message"]!= "<Media omitted>\n"]  # remove media_omttied
    f = open("stopwords.txt" ,'r')
    stop_words = f.read()
    stop_words = stop_words.split()

    words = []
    for msg in temp_data["message"]:
        for x in msg.lower().split():
            if x   not in stop_words:
                words.extend(x.split())

    new_data = pd.DataFrame(Counter(words).most_common(10))
    new_data = new_data.rename(columns= {0 : "Words" , 1:"Frequency"})
    return new_data


def emoji_filter(data,user):
    if(user != "Overall"):
        data =  data[data['User'] == user]
    emojis = []
    for msg in data["message"]:
            emojis.extend([c for c in msg if c in emoji.EMOJI_DATA])
    new_data = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return new_data
    

def monthly_timeline(data,user):
    if(user != "Overall"):
        data =  data[data['User'] == user]
    
    res = data.groupby(['year' , 'month'])['message'].count().reset_index()
    time = []
    for i in range(res.shape[0]):
        time.append(res["month"][i] + "-" + str(res["year"][i]))
    res["time"] = time
    return res

def daily_timeline(data,user):
    if(user != "Overall"):
        data =  data[data['User'] == user]
    
    daily_timeline = data.groupby('only_date')["message"].count().reset_index()
    return daily_timeline 

def week_activity(data,user):
    if(user != "Overall"):
        data =  data[data['User'] == user]
        if("count" in data.columns):
            data.rename(columns={'count': 'index'}, inplace=True)
    return data["day_name"].value_counts().reset_index()

def month_activity(data,user):
    if(user != "Overall"):
        data =  data[data['User'] == user]
    return data["month"].value_counts().reset_index()


def hours_activity(data,user):
    if(user != "Overall"):
        data =  data[data['User'] == user]
    return data.pivot_table(index = "day_name", columns="period", values = "message" ,aggfunc = 'count').fillna(0)


def logit_model(data,user):
    if(user != "Overall"):
        data =  data[data['User'] == user]
    
    data = data.dropna()
    X = data["message"]
    y = data["urgency"]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)

    model = DecisionTreeClassifier()
    model.fit(X.toarray(), y)

    return model,vectorizer


