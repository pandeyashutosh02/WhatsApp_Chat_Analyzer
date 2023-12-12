import streamlit as st 
import pandas as pd
import preprocessor,helper
import matplotlib.pyplot as plt
import seaborn as sns
import time

st.sidebar.title("Whatsapp Chat Analyzer")
st.sidebar.markdown("Please download your WhatsApp chat in a text file and upload it here.")
uploaded_file = st.sidebar. file_uploader("Choose a text file")
if uploaded_file is not None:
 
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocessor(data)

    # st.dataframe(df)

    #fetch unique user 
    user_list = df['User'].unique().tolist()
    if "Group Notification" in user_list:
        user_list.remove('Group Notification')
    user_list.sort()
    user_list.insert(0,"Overall")
    user_name = st.sidebar.selectbox("show analysis of" , user_list)
    if st.sidebar.button("Show Analysis "):

        # stast Area
        st.title("Top Statistics")
        total_msg, total_words,total_media,total_link = helper.stats(df,user_name)
        col1 ,col2,col3,col4 = st.columns(4)
        with col1:
            st.header("Total Messages")
            st.title(total_msg)
        with col2:
            st.header("Total Words")
            st.title(total_words)
        with col3:
            st.header("Total Media")
            st.title(total_media)
        with col4:
            st.header("Links Shared")
            st.title(total_link)


        # monthly timeline
        st.title("Monthly Timeline")
        data = helper.monthly_timeline(df ,user_name)
        fig,ax = plt.subplots()

        # plt.plot(data["time"] , data["message"] ,color = "green")
        plt.plot(data["time"].values, data["message"].values, color="green")
        plt.xticks(rotation = 90)
        plt.ylabel("Number of Messages")
        plt.xlabel("Month")
        plt.figure(figsize=(15,5))
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        data = helper.daily_timeline(df ,user_name) 
        fig,ax = plt.subplots()
        plt.plot(data["only_date"].values , data["message"].values,color = "orange")
        # plt.plot(data["only_date"] , data["message"],color = "orange")
        plt.xlabel("Date")
        plt.ylabel("Messages Frequency")
        plt.xticks(rotation = 90)
        st.pyplot(fig)

        # # Week Activity
        st.title("Activity Map")
        col1,col2 =st.columns(2)
        

        with col1:
            st.header("Most Busy Day")
            data = helper.week_activity(df,user_name)
            # st.dataframe(data)
            # st.write(data.columns)
            fig,ax = plt.subplots()
            ax.bar(data["day_name"].values , data["count"].values,color = "green")
            # ax.bar(data["index"] , data["day_name"],color = "green")
            plt.xlabel("Day")
            plt.ylabel("Frequency")
            plt.xticks(rotation = 90)
            st.pyplot(fig)

        with col2:
            st.header("Most Busy Month")
            data = helper.month_activity(df,user_name)
            fig,ax = plt.subplots()
            ax.bar(data["month"].values , data["count"].values,color = "green")
            # ax.bar(data["index"] , data["month"],color = "green")
            plt.xlabel("Month")
            plt.ylabel("Frequency")
            plt.xticks(rotation = 90)
            st.pyplot(fig)

        st.title("Most Busy Hours")
        fig,ax = plt.subplots()
        hours_data = helper.hours_activity(df,user_name)
        ax = sns.heatmap(hours_data)
        st.pyplot(fig)
        
        #finding busy users
        if user_name == "Overall":
            value,percetage_of_user = helper.busy_user(df)
            st.title("Most Busy Users")
            fig,ax = plt.subplots()
            col1 , col2 = st.columns(2)
            with col1:
                g = sns.barplot(x = value.index  ,y = value.values)
                plt.xticks(rotation= 90)
                st.pyplot(fig)
            with col2:
                    df2 = pd.DataFrame({'User':percetage_of_user.index,'Percent':percetage_of_user.values})
                    st.dataframe(df2)
        
        # st.title("WordCloud")  #wordcloud
        # st.subheader("Most Used Words By :  ")
        # st.write(user_name)
        # df_wc = helper.create_word_cloud(df ,user_name)
        # fig,ax = plt.subplots()
        # plt.axis('off')
        # ax.imshow(df_wc)
        # st.pyplot(fig)

        # Most common words
        st.title("Most Common Word")
        new_Data = helper.most_common_word(df ,user_name)
        # st.dataframe(new_Data)
        fig,ax = plt.subplots()
        sns.barplot(y = new_Data["Words"] , x = new_Data["Frequency"])
        # plt.xticks(rotation = 90)
        st.pyplot(fig)


        # emoji helper 
        # st.title("Emoji Filter")
        # data = helper.emoji_filter(df , user_name)
        # col1 ,col2 = st.columns(2)
        # with col1:
        #     st.dataframe(data)
        # with col2:
        #     fig, ax = plt.subplots()
        #     ax.pie( data[1].head(), labels = data[0].head(),autopct="%0.2f")
        #     st.pyplot(fig)
    st.sidebar.title("Machine Learning")
    message = st.sidebar.text_input('Enter a message to check wheather it is urgent or not:')
    # # helper.add_bg_from_url()
    if message:
        # st.write(message)
        model,vectorizer = helper.logit_model(df , user_name)
        message_features = vectorizer.transform([message])
        prediction = model.predict(message_features.toarray())
        new_Data = helper.most_common_word(df ,user_name)

        st.header("Machine Learning")  #(message in list(new_Data["Words"].head()))
        if ((prediction != 'non-urgent') or message in list(new_Data["Words"].head())) :
                st.write("You Entered : " , message)
                st.subheader("By analysing the Chats message thatyou have provided it seems that:")
                st.write('Message is urgent.')
        else:
                st.write("You Entered : " , message)
                st.subheader("By analysing the Chats message that you have provided it seems that:")
                st.write("Message is not urgent.")
