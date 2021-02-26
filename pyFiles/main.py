import csv,sys
import tweepy
from textblob import TextBlob


#---------------------------------------------------------------------------

inp = (sys.stdin.readlines()[0]).replace("[","")
inp = (inp).replace("]","")
inp = (inp).replace("\n","")
inp = ("".join(inp.split("\""))).split(",")

# authenticating
consumerKey = inp[2]
consumerSecret = inp[3]
accessToken = inp[4]
accessTokenSecret = inp[5]

auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)

api = tweepy.API(auth)

#-------------------------------------------------------------------------
search_tweet = inp[0]
    
t = []
tweets = api.search(search_tweet,count=20 tweet_mode='extended')
for tweet in tweets:
    polarity = TextBlob(tweet.full_text).sentiment.polarity
    subjectivity = TextBlob(tweet.full_text).sentiment.subjectivity
    t.append([tweet.created_at,tweet.full_text,tweet.user.id,tweet.user.location,polarity,subjectivity])
    

    
filename = 'pyFiles/dataset.csv'
testfields = ['created_at','full_text','user','location','polarity','subjectivity']

with open(filename, 'w') as csvfile:  
        # creating a csv writer object  
    csvwriter = csv.writer(csvfile)  
                
        # writing the fields  
    csvwriter.writerow(testfields)  
                
        # writing the data rows  
    csvwriter.writerows(t) 
