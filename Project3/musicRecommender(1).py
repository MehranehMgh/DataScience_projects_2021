# get arguments from command line
import sys
import os
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from collections import Counter

def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]


def remove_items(test_list, item):
    res = [i for i in test_list if i != item]
    return res

def main(args) -> None:
    """ Main function to be called when the script is run from the command line. 
    This function will recommend songs based on the user's input and save the
    playlist to a csv file.
    
    Parameters
    ----------
    args: list 
        list of arguments from the command line
    Returns
    -------
    None
    """
    """"
    arg_list = args[1:]
    if len(arg_list) == 0:
        print("Usage: python3 musicRecommender.py <csv file>")
        sys.exit()
    else:
        file_name = arg_list[0]
        if not os.path.isfile(file_name):
            print("File does not exist")
            sys.exit()
        else:
            userPreferences = pd.read_csv(file_name)

    # this code is just to check, delete later.
    #print(userPreferences.head())


    # 1. Use your train model to make recommendations for the user.
    # 2. Output the recommendations as 5 different playlists with
    #    the top 5 songs in each playlist. (5 playlists x 5 songs)
    # 2.1. Musics in a single playlist should be from the same cluster.
    # 2.2. Save playlists to a csv file.
    # 3. Output another single playlist recommendation with all top songs from all clusters.
    """
    mix = pd.read_csv(r'C:\Users\dp\Desktop\Semester7th\Projects\DataScience\3\mix.csv')
    #print(mix.head())
    #print(mix.shape)
    dataset = pd.read_csv(r'C:\Users\dp\Desktop\Semester7th\Projects\DataScience\3\final_dataset.csv')
    #print(dataset.head())
    cluster = pickle.load(open(r'C:\Users\dp\Desktop\Semester7th\Projects\DataScience\3\km.pickle', "rb"))
    mix_scld = StandardScaler().fit_transform(mix)
    #print(mix_scld.shape)
    pred = cluster.predict(mix_scld)
    #print(pred)
    test_pred = pred

    clusters = []
    for i in range(5) :
        NoOfCluster =most_frequent(test_pred)
        clusters.append(NoOfCluster)
        test_pred = remove_items(test_pred,NoOfCluster)

    #print(clusters)
    for i in range(5) :
        nameString = 'Daily Mix ' + str(i+1)
        playlist = dataset[dataset['Cluster'] == clusters[i]].sample(n=5, random_state=1)
        dailymix = pd.DataFrame(playlist)
        dailymix = dailymix.drop(['Cluster'], axis = 1)
        #print(dailymix)
        dailymix.to_csv(r'C:\Users\dp\Desktop\Semester7th\Projects\DataScience\3\ ' + nameString +'.csv', index=False)
    #print(dataset['Cluster'])



if __name__ == "__main__":
    # get arguments from command line
    args = sys.argv
    main(args)