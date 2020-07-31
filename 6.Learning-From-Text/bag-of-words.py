from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

string_1 = "Hi Lenon the self driving car will be late Best Jones"
string_2 = "Hi Jones the machine learning class will be great great great Best Lenon"
string_3 = "Hi Lenon the machine learning class will be most excellent"

# store them in a list
email_list = [string_1, string_2, string_3]

# fir data in bag of words
bag_of_words = vectorizer.fit(email_list)

# transformation
bag_of_words = vectorizer.transform(email_list)

print(bag_of_words)

# to check what 'feature number' a word is
print(vectorizer.vocabulary_.get("great"))
print(vectorizer.vocabulary_.get("machine"))