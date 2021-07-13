This was the dissertation project for a BSc in Computer Science at King's College London.

Please do have a look at the report  which will give an in-depth explaination. 



 Instructions to execute the code:

The following instructions will help in running the source code for the models built in this project. The source code is written in Python, and requires the installation of a few libraries.
• Setup an environment that can run Python3 code (Possibly an IDE).
• Before running the code, the libraries used in this project need to be downloaded. The libraries used in this project along are mentioned here. To install a library enter the associated command into a terminal window.
– scikit-learn : pip3 install -U scikit-learn – nltk: pip3 install -U nltk
– keras: pip3 install -U keras
– tensorflow: pip3 install tensorflow
– tweepy: pip3 install tweepy
– demoji: pip3 install demoji
– cleantext: pip3 install cleantext – pandas: pip3 install pandas
– numpy: pip3 install numpy
• Once all the libraries are installed, ensure you are in the code folder, and to download the GloVe embedding which is used here enter:
54
curl -O http://nlp.stanford.edu/data/glove.twitter.27B.zip.
This should download a zip file which then needs to be opened. Please ensure this file is placed in the source code folder.
• Once the GloVe embeddings have been downloaded, we can proceed to running the source code.
• To run the tweet extraction file, in the source code folder, enter python3 tweet extraction.py in the terminal. The code should execute, and display the extraction of data happening
in the terminal. The final extracted data will then be found in the tweets ext.csv file.
• To run the argument identification model built in this project, in the source code folder, enter python3 classify.py in the terminal. The code should execute, and display all results mentioned in the previous chapters.
• To run the argument component detection model built in this project, in the source code folder, enter python3 arg component det.py in the terminal. The code should execute, and display all results mentioned in the previous chapters.
