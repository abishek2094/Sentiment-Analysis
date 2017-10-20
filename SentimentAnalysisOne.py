from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

"""

This is a very simple implementation of Sentiment Analysis using the method of Random Forest Classification. This code was only
written to gain familiarity and understanding of very basic sentiment analysis and credits go to the individulas who procurred
the data and implemented the libraries used in this code as well as to the hosts who made available the tutorial/data.
The Bag Of Words approach is used to create the vocabulary and hence the features to the random forest classifier. This 
implementation consists of the following steps:

	1) Read the data set using the pandas library; The data set is obtained from Kaggle and consists of movie reviews; It can be
	   found at "https://www.kaggle.com/c/word2vec-nlp-tutorial/data". The data set is in a tab separated values (.tsv) format
       and consists of three attributes - id, sentiment (0 - bad, 1 - good) and review.
	
	2) Perform cleaning on the review data. This is composed of the following steps for each review,
		2.1) BeautifulSoup is used to remove unnecessary formatting such as <br \> from each review.
		2.2) All non-alphabet characters are removed from each review
		2.3) We use the nltk library to remove stop words, i.e., words that donot help us in the sentiment analysis from each review.
		
	3) Now, with the clean review data, features are constructed using the Bag of Words model using the CountVectorizer.
	
	4) The random forest classification model is now built using the scikit-learn library by feeding the features from the previous
	   step along with the sentiment data from the data set.
	   
	5) Now, repeat steps 1 to 3 to read and format the test data set. Once we have the clean test data, we use the model created in
	   the previous step to predict the sentiment of the test reviews.
	   
	6) Finally, the pandas library is used again to create a data frame to record and write the prediction results to a csv file.

"""


# A Global value that is used to hold the stop words that are to be removed from each review
stopWords = None

def readInputFromFile(filePath):
	"""
	This function uses the pandas library to read a dataset from a file given its path and returns the dataset it has read
	"""
	dataset = pd.read_csv(filePath, header=0, delimiter="\t", quoting=3)
	return dataset


def inputDataPreprocessing(inputData):
	"""
	This function is used to clean a review in the data set as described above and returns the cleaned review.
	"""
	global stopWords
	
	inputData = (BeautifulSoup(inputData, "html.parser")).get_text()
	inputData = re.sub("[^a-zA-Z]", " ", inputData.lower())
	words = inputData.split()
	meaningfulWords = [w for w in words if not w in stopWords]
	
	return (" ".join(meaningfulWords))


def trainTheModel():
	"""
	This function reads the training data, performs the cleanning of its reviews, constructs features using the cleaned reviews
	and finally creates the random forest classification model.
	"""

	trainingData = readInputFromFile("labeledTrainData.tsv")
	trainingDataSize = trainingData["review"].size

	cleanedTrainingReviews = []
	
	print("\n\nCleaning training data...")

	for i in range(0, trainingDataSize):
		if i % 100 == 0:
			print("\t",i+1, " of ", trainingDataSize, " completed !")
		cleanedTrainingReviews.append(inputDataPreprocessing(trainingData["review"][i]))


	vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
	trainDataFeatures = vectorizer.fit_transform(cleanedTrainingReviews)
	trainDataFeatures = trainDataFeatures.toarray()

	forestClassifier = RandomForestClassifier(n_estimators=100)
	forestClassifier = forestClassifier.fit(trainDataFeatures, trainingData["sentiment"])

	return [forestClassifier, vectorizer]
	
	
def testSetPrediction(forestClassifier, vectorizer):
	"""
	This function reads the test data, performs the cleanning of its reviews, constructs features using the cleaned reviews
	and finally predicts the sentiment of the reviews using the generated random forest classification model. These predictions
	are then written into a csv file.
	"""
	
	testData = readInputFromFile("testData.tsv")
	testDataSize = testData["review"].size
	
	cleanedTestData = []
	
	print("\n\nCleaning test data...")
	
	for i in range(0, testDataSize):	
		if i % 100 == 0:
			print("\t",i+1, " of ", testDataSize, " completed !")
		cleanedTestData.append(inputDataPreprocessing(testData["review"][i]))
	
	# We only tansform the test data to avoid overfitting.
	testDataFeatures = vectorizer.transform(cleanedTestData)
	testDataFeatures = testDataFeatures.toarray()
	
	classificationResult = forestClassifier.predict(testDataFeatures)
	
	output = pd.DataFrame(data = {"id":testData["id"], "sentiment":classificationResult})
	output.to_csv("ClassificationResult.csv", index=False, quoting=3)

	return
	
# Store the stopwords from nltk.corpus and store it as a set for faster execution.
stopWords = set(stopwords.words("english"))

# Train the model using the data and obtain the model as well as vectorizer
[forest, vect] = trainTheModel()

# Test the model on the test data.
testSetPrediction(forest, vect)
	
	