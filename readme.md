# Celebrity Image Classification Project

![](project_UI.png)

In this end-to-end data science and machine learning project, I classify sports personalities. After Hyperparameter tuning, I found that a Support Vector Machine (SVM) is best for this classification. The Model has 85% of accuracy in classifying images.In order to train the model I used images from google scraped with Fatkun chrome extention.

I also Restricted classification to only 5 people,
1) Maria Sharapova
2) Serena Williams
3) Virat Kohli
4) Roger Federer
5) Lionel Messi

### Folder structure

* <b>UI : This contains ui website code</b> 
* <b>server: Contains the Python flask server related code</b>
* <b>model: Contains python notebook for model building</b>

### Technologies used in this project,

* Python
* Numpy and OpenCV for data cleaning
* Matplotlib & Seaborn for data visualization
* Sklearn for model building
* Jupyter notebook, visual studio code as IDE
* Python flask for http server
* HTML/CSS/Javascript for UI
* Open-cv for manipulating images

### Required Libraries

* `PyWavelets`  0.5.2
* `opencv-python`  3.4.3.18
* `seaborn`  0.8.1
* `Flask`  1.0.2
* `numpy`  1.16.2
* `scikit-learn`  0.20.3

### Installation :

A good practice to start with a new project and use it, is to make a virtual enviornment for the particular project. Here is the steps for making virtual enviornment ::

1. `pip install virtualenv`
2. `python -m virtualenv myenv`

#### Install the dependencies of the App ::

Run commands on python terminal or anaconda terimial or any terminal you are using in your system.

* `pip install -r requirements.txt`

### Test the app:

* Clone the repository: `git clone https://github.com/izam-mohammed/celebrity_classification.git`
* Go to the project directory
* Go to Server Directory: `cd Server`
* Run the app: `python app.py`
* The development server will be up and running on port 5000 at the URL: http://127.0.0.1:5000/
* Now go to the UI Folder and open app.html on the browser. <b>Note that the flask app server must be up and running.</b> 
* Drag an image of your favourite celebrity from the five and hit the classify button. Our app will predict the celebrity name with his/ her image. It will also show us the percentage match of our image with all the five celebrities. 

Hope you like this project !!!
