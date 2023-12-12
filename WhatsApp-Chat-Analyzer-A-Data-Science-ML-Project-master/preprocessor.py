import re
import pandas as pd 

def preprocessor(data):
    pattern = "\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s"
    msg = re.split(pattern ,data)[1:]
    dates = re.findall(pattern,data)
    df = pd.DataFrame({"msg":msg , "date" : dates})
    df["date"] = pd.to_datetime(df["date"] , format = '%m/%d/%y, %H:%M - ')
    users = []
    msgs = []
    for messages in df["msg"]:
        entry = re.split('([\w\W]+?):\s',messages)
        if entry[1:]:
            users.append(entry[1])
            msgs.append(entry[2])
        else:
            users.append("Group Notification")
            msgs.append(entry[0])
    df["User"] = users
    df["message"] = msgs
    df.drop(columns= ["msg"],inplace = True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month_name()
    df["day"] = df["date"].dt.day
    df["hour"] = df["date"].dt.hour
    df["minute"] = df["date"].dt.minute
    df['month_num'] = df['month'].apply(lambda x: pd.to_datetime(x, format='%B').month)
    df["only_date"] =df["date"].dt.date
    df["day_name"] = df["date"].dt.day_name()
    period = []
    for hour in df['hour']:
        if hour == 23:
             period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-"+str(hour+1))
        else:
            period.append(str(hour) + "-"+str(hour+1))
    df["period"] =  period
    df.drop(columns= ["date"],inplace = True)
    df = df[df["message"] != "<Media omitted>\n"]
    
    urgent = []
    with open("urgent.txt" , 'r') as f:
        text = f.read()
    urgent.extend(text.split())
    text = " ".join(urgent)
    
    def urgency(msg):
        msg = str(msg)
        for x in msg.lower().split():
            if x in text or x == 'love':
                return "urgent"
        return "non-urgent"
    df["urgency"] = df["message"].apply(urgency)
    df = df.iloc[1:]
    return df
