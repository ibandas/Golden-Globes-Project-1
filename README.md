# gg-project-master
Golden Globe Project Master

Prerequistes:
- NLTK (https://www.nltk.org/install.html)
- Numpy (https://www.nltk.org/install.html)
- Fuzzywuzzy which requires python-levenshtein (https://www.geeksforgeeks.org/fuzzywuzzy-python-library/)
- Ensure that the data folder holds rpm_{year}.json where year is any year from 2010 up until 2020. These files are the IMDB json data files that consist of people in imdb matched with the titles of the movies they are known for. 
- Also ensure that the gg2013.json, gg2015.json, and gg2020.json files are in the data folder. Any data files should be of the form `gg[year].json`.

To run submission:
- Install dependencies using `pip3 install -r requirements.txt`. Also use the links above to install the dependencies.
- In console, type "python3 autograder.py {year}", year being the year of data you want to run it on.

Note:
- It will take time to download for the first time libraries from NLTK, such as stopwords, names, and more. This simply can be downloaded by running the gg_api.py file for the first time, or simply the autograder. Please take this into consideration when focusing on the time constraints of the assignment.
- Also, ensure that everything is always run in python3 because it may cause issues with encoding/decoding ascii. A simple command would be to do an "alias python=python3" so that you don't have to type python3 each time.
- Also, note that if you are running in python3, you want to install the dependencies with pip3 instead of pip to make sure python3 can reach the dependencies.

Human readable output will be sent to the console.
Repository: https://github.com/ibandas/Golden-Globes-Project-1
