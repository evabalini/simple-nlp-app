# simple-nlp-app
A simple NLP app made with FastAPI that trains a model to predict movie categories based on their description.

To run the app run: univorn --reload api:app
To post a csv, go to http://127.0.0.1:8000/docs#/ (or where you serve the app) to see the SwaggerUI interface. 

To train:
- Click on the POST /train method
- Click on the "Try it out" button
- Select the .csv file containing the training set. 
- Click on "execute"

To test:
- Do the same as above but in the POST /test method and with the test.csv file. 
