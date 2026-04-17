# import required modules
import spacy
nlp = spacy.load("en_core_web_sm")
  
# taking input
text = "EMIRATES"
  
# returns a document of object
doc = nlp(text)
  
# checking if it is a noun or not
if(doc[0].tag_ == 'NNP'):
    print(text, " is a noun.")
else:
    print(text, " is not a noun.")