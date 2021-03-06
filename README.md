## ML-Model-Flask-Deployment
This is a demo project to elaborate how Machine Learn Models are deployed on production using Flask API

### Prerequisites
You must have Scikit Learn, Pandas (for Machine Learning Model) and Flask (for API) installed.

### Project Structure
This project has three major parts :
1. model.py - This contains code fot our Machine Learning model to predict Daraz's product score based on training data in 'Extension Data.csv' file.
2. app.py - This contains Flask APIs that receives employee details through GUI or API calls, computes the predicted value based on our model and returns it.
3. templates - This folder contains the HTML template to allow user to enter product detail and displays the predicted Daraz product score.

### Running the project
1. Ensure that you are in the project home directory. Create the machine learning model by running below command -
```
python model.py
```
This would create a serialized version of our model into a file model.pkl

2. Run app.py using below command to start Flask API
```
python app.py
```
By default, flask will run on port 5000.

3. Navigate to URL http://localhost:5000

You should be able to view the homepage as below :
![alt text](https://ahsanasgharfyp.herokuapp.com/)

Enter valid numerical values in all 3 input boxes and hit Predict.

If everything goes well, you should  be able to see the predicted score value on the HTML page!
![alt text](https://ahsanasgharfyp.herokuapp.com/predict)

4. You can also send direct POST requests to FLask API using Python's inbuilt request module
Run the below command to send the request with some pre-populated values -
```
address of the app </calculate>
For example, localhost:5000/calculate
```
