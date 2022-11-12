# %%
# %pip install pyabsa

# %%
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from pyabsa.functional import ATEPCCheckpointManager
from pathlib import Path
from pyabsa import available_checkpoints
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials
from flask import Flask

# %%
from fastapi import FastAPI


# %%

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

# %%

credpath = {
    "type": "service_account",
    "project_id": "radar-application-1488d",
    "private_key_id": "f6cb1dc14b3dc38046b0fb210b2dfd3d13a28492",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCXpLPG47dSuefI\nQilyR4oBYz5AxBbFhc4itQa4ybmcFF93+bs97ZUW/LM1YNCdb76Mlkgdtyfb3P4u\nbk9gZ6RAqQLnO90tw+bHtxX6Q29Rh1LuecgSqqZZrqdrSBcriBaVHzdBT1t914S4\nTcb2gpmUj+596z9S/SH3NC20KvNFo/Ai3/UmgwdcNIasKpBQrQASJfP7wU1gHf7T\nvnyZnexFUX2BmsmUEkq7RZfcii+98kQQ/zv+CgxU1SkSXFY3WCLag0XAlsGIxVyd\ncLOSs9PnEzi08m5LYIPsnfzqa6KnYv34qnXoOGRTSk5ZdPRrzLxvNBJNRuSN0wOk\n9M39Dky/AgMBAAECggEAPSTK3EILNA8DlyqePZb83Uxf2It4RxKJqFLnr/Cep4FL\ncTu/tNusBsXDmJ094I0i/trFnz2vk6ZK0vvlg5CmmO/M3OG1b/OShSqccPlp1CzF\nUqTF+EjYpEaY+Nfrh8DqohwhEnNmB5qzyACMXe8Q7+cNGbaWJOcuH9fpKcE7r/Mn\nyQtzhRdKu+sUSr2+w1jJ66naKQvlfRkQZyAs1NE1lDTlGNmHKe9cMXV0iVuZyKRq\nbAiIxjyCB/NMObwq2trPJ0mhRQdc9d0PJFHMaOKBEhhS3YdDbgIMyj7rUMo4alI8\nNiawsskLq9D6DQNrzFdp33ewB21eNTshkhpXxwuQGQKBgQDVMI4jqliPFYpKmooO\n+Zx3ggLksslDmDxaIs2gjReuvvLXZYBbG7SKpMtUsWZPYs5C5p8nW7GRRfCArmVo\nUwXd0PZwt4sPg0HyoflAlliB/kecr1bVzWfH2ogTySguLW+tWV/zYRAMN2FoEPZR\ntkFwWHJyJduA5JvxAo+nMH4FvQKBgQC2GDw/b64edVyplNkJz6mgiTPjnQn+g5FC\nQua5/xPLaIbXMMzNlxTFXBZ6ZoPreao6dG3jOAxj4Wpd9XY2JKvD9MgWRrjEmbyG\nA93Snfgn9w/Iaa51TanjPExSUqWNNrh011X9ynYQn8YVOHxhUZ9FeSjYIQMTHGFf\nIvjuK48OKwKBgQCMz6hiqEYcI/cWtaJQp9AQI4BzvB8xlWDvjCNTUz38PsU5PiKc\nit0h4h0nEJFqB/ICwD8JCQhs0sw6wnXahVPPohDUfHbORT0O3Ks8XNGS8vgr5qgt\nSaGtoIrWvrvaXEpyLiExKMAnwYCF8wYvDHmGkfTtrlGgfd7+PlnR7Taf5QKBgBhZ\nzVS+Xo58K1QSL6P8PTbWojXB/mAmv/oYcDpXPhJpe/6y6/BiT8jEs8zSgLmwn28J\nutgz2pRQxKSj+pbq+H1P8qHn+zVvSaKySausrE7L3zRxzX6qUBmvKpWnr7PeqXQW\nh81Ukc1PUHHuB9QL0jy8IxYj9AFOPkc2qgtPj+XZAoGBAKqTRxgt8XKx7ewgq/c2\n/h38i3H/NCiLy3ETkMiOWJYYUApYfqdgeLyvXqVRR008xYTIGDqF63lmkut2leH1\nBneR3uOIBfxp/jIescqqayiQN1198e4anmxwOOJyEm8WNZ4B7aLJRsqVFoq9UPxC\nzDNN6bA/pCG7i+AwQ9479KAm\n-----END PRIVATE KEY-----\n",
    "client_email": "firebase-adminsdk-gh7k5@radar-application-1488d.iam.gserviceaccount.com",
    "client_id": "108252273274804916220",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-gh7k5%40radar-application-1488d.iam.gserviceaccount.com"
}

if not firebase_admin._apps:
    login = credentials.Certificate(credpath)
    default_app = firebase_admin.initialize_app(login)

# %%
db = firestore.client()
feedbacks = db.collection("volunteer_feedbacks").order_by(
    u'CreatedOn', direction=firestore.Query.DESCENDING).limit(1).stream()

# %%
sentences = []
for feedback in feedbacks:
    feedbackOnly = feedback.to_dict()
    sentences.append(feedbackOnly)

print(sentences)

# %%
checkpoint_map = available_checkpoints()

# %%

aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(
    checkpoint='english', auto_device=True)

# %%
feedback = []

sen = sentences[-1].get("Feedback")
feedback.append(sen)
feedback

# %%
volunteerId = []

sen = sentences[-1].get("Volunteer ID")
# volunteerId.append(sen)
# volunteerId
print(sen)

# %%
examples = feedback
inference_source = examples
atepc_result = aspect_extractor.extract_aspect(inference_source=inference_source,
                                               save_result=True,
                                               print_result=True,  # print the result
                                               pred_sentiment=True,  # Predict the sentiment of extracted aspect terms
                                               )

# %%
aspect = atepc_result[0].get("aspect")
aspect[0]

# %%
confidence = atepc_result[0].get("confidence")
rating = (confidence[0]*5)
rating

# %%
atepc_result

# %%
sentiment = atepc_result[0].get("sentiment")
sentiment[0]

# %%

complete_score_pos = []
complete_score_neg = []
complete_score = []
polarity_score = []
subjectivity_score = []
for sentiment in feedback:
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment)
    neg = score['neg']
    pos = score['pos']
sentiment_score = score.get("pos") - score.get("neg")

print(sentiment_score)

# %%
# db.collection("aspect").add({'aspect': aspect, 'sentiment': sentiment})

docs = db.collection('volunteer_feedbacks').order_by(
    u'CreatedOn', direction=firestore.Query.DESCENDING).limit(1).stream()
for doc in docs:
    key = doc.id
    db.collection('volunteer_feedbacks').document(key).update({'aspect': aspect, 'sentiment_score': sentiment_score,
                                                               'sentiment': "Positive" if sentiment_score > 0 else "Negative"})

# %%
# docs = db.collection('test_volunteers').order_by(u'CreatedOn', direction=firestore.Query.DESCENDING).limit(1).stream()
docs = db.collection('users').get()
for doc in docs:
    if doc.to_dict()['uid'] == sen:
        key = doc.id
        currentRating = doc.to_dict()['volunteerRating']
        cumilativeRating = sentiment_score/20 + currentRating
        db.collection('users').document(key).update(
            {'volunteerRating': round(cumilativeRating, 2)})

# %%


@app.get("/successful")
async def root():
    return {"Volunteer Rating": "Successful"}
