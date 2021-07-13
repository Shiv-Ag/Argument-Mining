
import tweepy
import json
import csv
import demoji
auth_data = json.load(open('credentials.json'))

demoji.download_codes()

#Authentication setup for Twitter API
auth = tweepy.AppAuthHandler(auth_data['CONSUMER_KEY'], auth_data['CONSUMER_SECRET'])
#auth = tweepy.OAuthHandler(auth_data['CONSUMER_KEY'], auth_data['CONSUMER_SECRET'])
#auth.set_access_token(auth_data['ACCESS_KEY'], auth_data['ACCESS_SECRET'])

api = tweepy.API(auth)


i = 0
tweepy.debug(True)

#Using cursoring to join pages of extracted tweets

r = tweepy.Cursor(api.search, q = 'vaccine', count = 100, lang ='en', tweet_mode = 'extended').items()

#Storing the extracted in a csv file

with open('tweets_ext.csv','w') as f1:
  writer=csv.writer(f1, lineterminator="\n")
  for tweet in r:
   print(type(tweet))
   i = i+1
   try:
    res = tweet.retweeted_status.full_text.encode('utf-8', 'ignore')
   except AttributeError:  # Not a Retweet
    res= tweet.full_text.encode('utf-8', 'ignore')
   label = (str(i))
   res = demoji.replace(string=((res).decode('utf-8')), repl = '')
   writer.writerow([label, res]),
print(i)


tweepy.debug(True)
