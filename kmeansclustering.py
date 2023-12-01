import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
"""

Pour evaluer la qualité du clustering:
WCSS ==> ∑i=1 k  ∑j=1 ni ∥xij − ci∥^2

"""

def load_csv(path):
    return np.genfromtxt(path, delimiter=',', skip_header=1)
    
class KmeansClustering: # todo: elbow methode
    def __init__(self, max_iter=10, k=5):
        self.k = k # number of centroids (cluster)
        self.max_iter = max_iter # number of max iterations to update the centroids
        self.centroids = [] # values of the centroids
        self.dim = 3
        
        
    def euclidiean_methode(self, data_points):
        return np.sqrt(np.sum((self.centroids - data_points)**2, axis=1))
    
    # Fonction calculer_wcss(W, centroids, assignments):
    # Calculer la somme des carrés intra-cluster (WCSS)
    # retourner la somme des carrés des distances euclidiennes entre chaque point et le centroïde de son cluster

    # Fonction trouver_coude(wcss_values):
    # Trouver le coude en utilisant la méthode du coude (Elbow method)
    # Retourner la valeur de k au coude
    
    
    def fit(self, X):
        print(X.shape)
        self.centroids = np.random.uniform(np.nanmin(X, axis=(0, 1)), np.nanmax(X, axis=(0, 1)), size=(self.k, self.dim))
                                            
        for _ in range(self.max_iter): # and WCSS (change pas)
            clusters_ids = []
            res = 0
            for data_vetcor in X: # calcul distance pour chque vecteur (L2 et ensuite similarite cosinus)
                distance = self.euclidiean_methode(data_vetcor) # distance entre ce point et tous les centroids
                cluster_id = np.argmin(distance) # savoir quel est le centroide le plus proche pour ce point
                clusters_ids.append(cluster_id)
            
            clusters_ids = np.array(clusters_ids) # le nb du cluster de chaque vecteur

            clusters_indexes = []
            for i in range(self.k):
                clusters_indexes.append(np.argwhere(clusters_ids == i)) # on creer trois liste ou l'ont regroupe les vecteurs d'un meme cluster
            
            
            # faire la moyenne de tous les points du cluster et mettre le new centroide ici
            clusters_centers = []
            for i, indice in enumerate(clusters_indexes):
                if len(indice) == 0:
                    clusters_centers.append(self.centroids[i])
                else:
                    clusters_centers.append(np.mean(X[indice], axis=0)[0])
             # MAJ des centroids      
            if np.max(self.centroids - np.array(clusters_centers)) < 0.001:
                break
            else:
                self.centroids = np.array((clusters_centers))



                #'''
            #DISPLAY:  
                fig = plt.figure(figsize=(10, 6))
                
            # -- 2D -- 
                # fig, ax = plt.subplots(figsize=(10, 6))
    
            # Affichage des points dans l'espace 3D
                ax = fig.add_subplot(111, projection='3d')
                #ax.scatter(X[:, :, 0], X[:, :, 1], X[:, :, 2], c=clusters_ids)
                ax.scatter(self.centroids[:, :, 0], self.centroids[:, :, 1], self.centroids[:, :, 2],
                           c=range(len(self.centroids)), cmap='plasma', marker='x', s=200)
    
                ax.set_xlabel('height')
                ax.set_ylabel('weight')
                ax.set_zlabel('bone_density')
                
                '''
            # Change POV:
                elevation_angle = 20  # Angle d'élévation
                azimuth_angle = 30  # Angle azimutal
                ax.view_init(elev=elevation_angle, azim=azimuth_angle)
                '''
                
                plt.show()
                
                time.sleep(0.5)
                #'''
                
            
        return clusters_ids




def main():
    kmeans = KmeansClustering()
    csv_values = load_csv("dataset.csv")
    X = np.array(csv_values)
    selected_columns = X[:, 1:]

    # Création de la matrice de vecteurs
    result_matrix = np.reshape(selected_columns, (len(X), -1, 3))

    kmeans.fit(result_matrix)
    

if __name__ == "__main__":
    main()