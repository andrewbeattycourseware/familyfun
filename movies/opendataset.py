import pandas as pd
import codecs


DATA_DIR = "./ml-100k/"
INFO_FILE = "u.info"
GENRE_FILE = "u.genre"
USER_FILE = "u.user"
ITEM_FILE = "u.item"
OCCUPATION_FILE = "u.occupation"
#DATA_FILE = "u.data"
DATA_FILE = "u6.test"

def readinfo():
    with open(DATA_DIR + INFO_FILE, "r") as f:
        # first line is number of users
        line = f.readline() 
        # extract the number of users ie number before the space
        num_user = int(line.split(" ")[0]) 
        # next line is items
        line = f.readline()
        num_items = int(line.split(" ")[0])
        line = f.readline()
        num_ratings = int(line.split(" ")[0])
        return (num_user, num_items, num_ratings)
# test code
#num_user, num_items, num_ratings = readinfo()
#print(num_user, num_items, num_ratings)

def readusers(num):
    header_names = ['user id', 'age', 'gender', 'occupation', 'zip code']
    # the user file is not a tab separated list the items are seperted with a |
    df = pd.read_csv(DATA_DIR+USER_FILE, sep="|", names=header_names, nrows=num)
    #print(df.head())
    return df

def readitems(num):
    header_names = ["movie id","movie title", "release date","video release date","IMDb URL","unknown","Action","Adventure","Animation","Children's","Comedy","Crime","Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western" ]
    # the user file is not a tab separated list the items are seperted with a |
    # I have had to delete the last blank line from the dataset
    df = pd.read_csv(DATA_DIR+ITEM_FILE, sep="|", names=header_names,  nrows=num)
    #print(df.head())
    return df


def readdata():
    header_names = ['user id', 'item id',  'rating', 'timestamp']
    # the user file is not a tab separated list the items are seperted with a |
    # I have had to delete the last blank line from the dataset
    df = pd.read_csv(DATA_DIR+DATA_FILE, names=header_names, sep="\t")
    #print(df.head())
    return df


# test code
num_user, num_items, num_ratings = readinfo()
df_user = readusers(num_user)
#print(num_user, num_items, num_ratings)
df_items = readitems(num_items)
df_data = readdata()
#print (df.head())
