# :star2: Celebrity Image Classification Project :camera:

<p>
<img src="https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=white" />
<img src="https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white" />
<img src="https://img.shields.io/badge/Python-239120?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/css3-1572B6?logo=css3&logoColor=white" />
<img src="https://img.shields.io/badge/numpy-%23013243.svg?logo=numpy&logoColor=white"/>
<img src="https://img.shields.io/badge/GIT-E44C30?logo=git&logoColor=white" />
<img src="https://img.shields.io/badge/prettier-1A2C34?logo=prettier&logoColor=white" />
<img src="https://img.shields.io/badge/GitHub_Actions-563D7C?logo=github-actions&logoColor=white"/>
<img src="https://img.shields.io/badge/html5-E34F26?logo=html5&logoColor=white" />
</p>

In this end-to-end data science and machine learning project, I classify sports personalities. After Hyperparameter tuning, I found that a Support Vector Machine (SVM) is best for this classification. The Model has chart_with_upwards_trend: 85% accuracy in classifying images. In order to train the model, I used images from Google scraped with Fatkun Chrome extension.

![](https://github.com/izam-mohammed/Celebrity-Classification/blob/main/project_UI.png?raw=true)

### :busts_in_silhouette: Restricted classification to only 5 people:
1) Maria Sharapova
2) Serena Williams
3) Virat Kohli
4) Roger Federer
5) Lionel Messi

### :file_folder: Folder structure
* **UI :** This contains UI website code
* **server:** Contains the Python Flask server-related code
* **model:** Contains Python notebook for model building

### :computer: Technologies used in this project
* Python
* Numpy and OpenCV for data cleaning
* Matplotlib & Seaborn for data visualization
* Sklearn for model building
* Jupyter Notebook, Visual Studio Code as IDE
* Python Flask for HTTP server
* HTML/CSS/JavaScript for UI
* OpenCV for manipulating images

### :books: Required Libraries
* `PyWavelets` 0.5.2
* `opencv-python` 3.4.3.18
* `seaborn` 0.8.1
* `Flask` 1.0.2
* `numpy` 1.16.2
* `scikit-learn` 0.20.3

### :wrench: Installation
A good practice to start with a new project and use it is to make a virtual environment for the particular project. Here are the steps for making a virtual environment:
1. `pip install virtualenv`
2. `python -m virtualenv myenv`

#### Install the dependencies of the App
Run commands on the Python terminal or Anaconda terminal or any terminal you are using in your system:
* `pip install -r requirements.txt`

### :rocket: Test the app
1. Clone the repository: `git clone https://github.com/izam-mohammed/celebrity_classification.git`
2. Go to the project directory
3. Go to Server Directory: `cd Server`
4. Run the app: `python app.py`
5. The development server will be up and running on port 5000 at the URL: http://127.0.0.1:5000/
6. Now go to the UI Folder and open app.html on the browser. <b>Note that the Flask app server must be up and running.</b>
7. Drag an image of your favorite celebrity from the five and hit the classify button. Our app will predict the celebrity's name with his/her image. It will also show us the percentage match of our image with all the five celebrities.

Hope you like this project !!! :thumbsup:
