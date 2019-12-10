from imdb import IMDb
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
import os.path
from wordcloud import WordCloud
import matplotlib.pyplot as plt 

imdb=IMDb()

def contents_id(contents): 
    contents_id_list=[]
    for i in contents:
        content=imdb.search_movie(i)
        cont_id=content[0].movieID
        contents_id_list.append(cont_id)
    return contents_id_list

def title(content_id):
    content=imdb.get_movie(content_id)
    title_name=str(content['title'])
    return title_name

def genre(content_id):
    content=imdb.get_movie(content_id)
    genres=content['genres']
    genres=" ".join(str(i) for i in genres)
    return genres
    
def key_words(content_id):
    content=imdb.get_movie(content_id)
    plot=content['plot'][0]
    punctuations='''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for i in plot:
        if i in punctuations:
            plot=plot.replace(i,"")
    r=Rake()
    r.extract_keywords_from_text(plot)
    key_words_dict_scores=r.get_word_degrees()
    keyword=list(key_words_dict_scores.keys())
    keyword=" ".join(str(i) for i in keyword)
    return keyword
    
def director(content_id):
    content=imdb.get_movie(content_id)
    dir_name=[]
    for director in content['directors']:
        dir_name.append(director['name'])
    dir_name=" ".join(str(i) for i in dir_name) 
    return dir_name
    
def cast(content_id):
    content=imdb.get_movie(content_id)
    cast_name=[]
    for cast in content['cast']:
        cast_name.append(cast['name'])
    cast_name=" ".join(str(i) for i in cast_name)
    return cast_name

def get_title_from_index(df,index):
    return df[df.index==index]["title"].values[0]

def get_index_from_title(df,title_content):
    return df[df.title==title_content]["index"].values[0]

def recommendation_algorithm(df,title_content):
    tf=TfidfVectorizer()
    count_matrix=tf.fit_transform(df["combined_features"])
    cosine_sim=cosine_similarity(count_matrix)
    title_user_likes=title_content
    title_index=get_index_from_title(df,title_user_likes)
    similar_titles=list(enumerate(cosine_sim[title_index]))
    sorted_similar_titles=sorted(similar_titles,key=lambda x:x[1],reverse=True)[1:]
    i=0
    title_list=[]
    for element in sorted_similar_titles:
        title_list.append(get_title_from_index(df,element[0]))
        i+=1
        if i>=5:
            break
    return title_list

def wordcloud(genres):
    genres=" ".join(str(i) for i in genres)
    wordcloud=WordCloud(width=800,height=800,background_color='white',min_font_size=10).generate(genres) 			 
    plt.figure(figsize=(8, 8)) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    plt.show()

def recommended_titles_genre_filter(genres,genre_filter):
    if(genre_filter in genres):
        return True
    else:
        return False

def genre_filter(recommended_titles,genres):
    genres_filter=" ".join(str(i) for i in genres)
    genres_filter=genres_filter.split()
    genres_list=["Action","Adventure","Animation","Biography","Comedy","Crime","Drama","Fantasy",
                 "Historical","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War"]
    print("Please select your prefered Genre:\n")
    for i in range(len(genres_list)):
        print((i+1),"-",genres_list[i])
    while(True):
        c=(int)(input("Please enter your choice: "))
        if(c>=1 and c<=16):
            if(genres_list[c-1] in genres_filter):
                break
            else:
                print("You have choosen the Genre, which is not in our Recommended Titles Genres list")
        else:
            print("Invalid option")
    positions=[]        
    filtered_titles=[]
    for i in range(len(genres)):
        if(recommended_titles_genre_filter(genres[i],genres_filter[c-1])==True):
            positions.append(i)
    for i in positions:
        filtered_titles.append(recommended_titles[i])
    print("The Titles as per the Genre: ",genres_list[c-1])
    print(filtered_titles)
    print('\nDo you want to save the Recommended Filtered Titles to a ".txt" file')
    print("Please select 1 for Yes")
    print("              2 for No")
    while(True):
        c=(int)(input("Please enter your choice: "))
        if(c>=1 and c<=2):
            break
        else:
            print("Invalid option")
    if(c==1):
        print("Please choose the folder where you want to save the Recommended Filtered Titles list:")
        folder=askdirectory()
        file_name=input("Please enter the File Name: ")
        file_name=file_name+".txt"
        complete_path=os.path.join(folder,file_name)
        recommended_file=open(complete_path,'w')
        recommended_file.write(recommended_titles)
        recommended_file.close()
    
def movies_recommendation(train_content_id_list,test_content_id_list):
    titles=[]
    genres=[]
    directors=[]
    casts=[]
    keywords=[]
    recommended_titles=[]
    for i in test_content_id_list:
        titles.append(title(i))
        genres.append(genre(i))
        directors.append(director(i))
        casts.append(cast(i))
        keywords.append(key_words(i))
    for i in train_content_id_list:
        title_content=title(i)
        titles.append(title(i))
        genres.append(genre(i))
        directors.append(director(i))
        casts.append(cast(i))
        keywords.append(key_words(i))
        df=pd.DataFrame({"genres":genres,"title":titles,"keywords":keywords,"directors":directors,"casts":casts})
        df=df[['title','genres','keywords','directors','casts']]
        df["combined_features"]=df['keywords']+" "+df['casts']+" "+df["genres"]+" "+df["directors"]
        df.reset_index(inplace=True)
        recommended_titles.append(recommendation_algorithm(df,title_content))
        titles=titles[:-1]
        genres=genres[:-1]
        directors=directors[:-1]
        casts=casts[:-1]
        keywords=keywords[:-1]
    recommended_titles=np.array(recommended_titles)
    recommended_titles=recommended_titles.flatten()
    recommended_titles=recommended_titles.tolist()
    recommended_titles=list(dict.fromkeys(recommended_titles))
    print(recommended_titles[:5])
    recommended_titles_copy=recommended_titles
    recommended_titles_genres=[]
    for i in recommended_titles:
        recommended_titles_genres.append(genre(i))
    recommended_titles=" ".join(str(i) for i in recommended_titles) 
    print('\nDo you want to save the Recommended Movies to a ".txt" file')
    print("Please select 1 for Yes")
    print("              2 for No")
    while(True):
        c=(int)(input("Please enter your choice: "))
        if(c>=1 and c<=2):
            break
        else:
            print("Invalid option")
    if(c==1):
        print("\nPlease choose the folder where you want to save the Recommended Movies list:")
        folder=askdirectory()
        file_name=input("Please enter the File Name: ")
        file_name=file_name+".txt"
        complete_path=os.path.join(folder,file_name)
        recommended_file=open(complete_path,'w')
        recommended_file.write(recommended_titles)
        recommended_file.close()
        print("Your file has been saved by the name: ",file_name)
    wordcloud(recommended_titles_genres)
    print("\nDo you want to Filter the Recommended Movies by Genre")
    print("Please select 1 for Yes")
    print("              2 for No")
    while(True):
        c=(int)(input("Please enter your choice: "))
        if(c>=1 and c<=2):
            break
        else:
            print("Invalid option")
    if(c==1):
        genre_filter(recommended_titles_copy,recommended_titles_genres)

def series_recommendation(train_content_id_list,test_content_id_list):
    titles=[]
    genres=[]
    keywords=[]
    recommended_titles=[]
    for i in test_content_id_list:
        titles.append(title(i))
        genres.append(genre(i))
        keywords.append(key_words(i))
    for i in train_content_id_list:
        title_content=title(i)
        titles.append(title(i))
        genres.append(genre(i))
        keywords.append(key_words(i))
        df=pd.DataFrame({"genres":genres,"title":titles,"keywords":keywords})
        df=df[['title','genres','keywords']]
        df["combined_features"]=df['keywords']+" "+df["genres"]
        df.reset_index(inplace=True)
        recommended_titles.append(recommendation_algorithm(df,title_content))
        titles=titles[:-1]
        genres=genres[:-1]
        keywords=keywords[:-1]
    recommended_titles=np.array(recommended_titles)
    recommended_titles=recommended_titles.flatten()
    recommended_titles=recommended_titles.tolist()
    recommended_titles=list(dict.fromkeys(recommended_titles))
    print(recommended_titles)
    recommended_titles_copy=recommended_titles
    recommended_titles_genres=[]
    for i in recommended_titles:
        recommended_titles_genres.append(genre(i))
    recommended_titles=" ".join(str(i) for i in recommended_titles)
    print('\nDo you want to save the Recommended TV Series to a ".txt" file')
    print("Please select 1 for Yes")
    print("              2 for No")
    while(True):
        c=(int)(input("Please enter your choice: "))
        if(c>=1 and c<=2):
            break
        else:
            print("Invalid option")
    if(c==1):  
        print("\nPlease choose the folder where you want to save the Recommended TV Series list:")
        folder=askdirectory()
        file_name=input("Please enter the File Name: ")
        file_name=file_name+".txt"
        complete_path=os.path.join(folder,file_name)
        recommended_file=open(complete_path,'w')
        recommended_file.write(recommended_titles)
        recommended_file.close()
        print("Your file has been saved by the name: ",file_name)
    wordcloud(recommended_titles_genres)
    print("\nDo you want to Filter the Recommended TV Series by Genre")
    print("Please select 1 for Yes")
    print("              2 for No")
    while(True):
        c=(int)(input("Please enter your choice: "))
        if(c>=1 and c<=2):
            break
        else:
            print("Invalid option")
    if(c==1):
        genre_filter(recommended_titles_copy,recommended_titles_genres)

def main():
    print("Welcome to Movies/TV Series Recommender....!!!!!")
    print("\nYou have to choose 2 Text Files: Train and Test")
    print("Train FIle will contains the Titles which you had already watched")
    print("Test FIle will contains the Titles which you want to watch")
    print("\nPlease write every Title in a new line, and there shouldn't be any new line after the Last Title")
    print("\nPlease enter 1 for Movies")
    print("             2 for TV Series")
    while(True):
        c=(int)(input("Please enter your choice: "))
        if(c==1 or c==2):
            break
        else:
            print("Invalid option")
    print("\nPlease choose the Test File:")
    while(True):
        path_test=askopenfilename()
        extension=path_test[-4:]
        if(extension==".txt"):
            break
        else:
            print('You have not choosen a ".txt" file')
    test_file=open(path_test,"r")
    test_file_list=test_file.read().split("\n")
    test_content_id_list=contents_id(test_file_list)
    print("\nPlease choose the Train File:")  
    while(True):
        path_train=askopenfilename()
        extension=path_train[-4:]
        if(extension==".txt"):
            if(path_train!=path_test):
                break 
            else:
                print("You have choosen the same file")
        else:
            print('You have not choosen a ".txt" file')   
    train_file=open(path_train,"r")
    train_file_list=train_file.read().split("\n")
    train_content_id_list=contents_id(train_file_list)
    if(c==1):
        movies_recommendation(train_content_id_list,test_content_id_list)
    elif(c==2):
        series_recommendation(train_content_id_list,test_content_id_list)

main()
