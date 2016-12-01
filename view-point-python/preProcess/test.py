import xtractFeatures
import clustering

dbPath = '/media/Data/Flickr-YsR/merlionImages/'
file_name = "feature.list"

n_clusters = 5

def xtractFeatures(dbPath):
    obj = xtractFeatures.xtractFeatures(sPath=dbPath, rgb=True, surf=True, orb=False, hog=True)
    obj.xtract()    


def cluster(dbPath, file_name, n_clusters):
    obj = clustering.clustering(db_path=dbPath, file_name=file_name, n_clusters=n_clusters)    
#xtractFeatures(dbPath)    

cluster(dbPath, file_name, n_clusters)    
