import Data
import Evaluate
import sklearn.cluster as cluster
import numpy as np
from keras.models import load_model


def flatten_data(data):
    return np.array([record.flatten() for record in data])


print("KMeans: ",
      Evaluate.evaluate(
          cluster.KMeans(n_clusters=5),
          Data.X_1_, Data.y_1_))

print("Agglomerative: ",
      Evaluate.evaluate(
          cluster.AgglomerativeClustering(
              n_clusters=5, linkage="ward"),
          Data.X_1_, Data.y_1_))


X_1_flatten = flatten_data(Data.X_1)

print("KMeans on ts: ",
      Evaluate.evaluate(
          cluster.KMeans(n_clusters=5),
          X_1_flatten, Data.y_1))

print("Agglomerative on ts: ",
      Evaluate.evaluate(
          cluster.AgglomerativeClustering(
              n_clusters=5, linkage="ward"),
          X_1_flatten, Data.y_1))

encoded = load_model("encoder_simple_relu.h5").predict(Data.X_1_)

print("KMeans on Encoded: ",
      Evaluate.evaluate(
          cluster.KMeans(n_clusters=5),
          encoded, Data.y_1_))

print("Agglomerative on Encoded: ",
      Evaluate.evaluate(
          cluster.AgglomerativeClustering(
              n_clusters=5, linkage="ward"),
          encoded, Data.y_1_))

encoded_ts = flatten_data(load_model("encoder_conv_relu.h5").predict(Data.X_1))


print("KMeans on Encoded with ts: ",
      Evaluate.evaluate(
          cluster.KMeans(n_clusters=5),
          encoded_ts, Data.y_1))

print("Agglomerative on Encoded with ts: ",
      Evaluate.evaluate(
          cluster.AgglomerativeClustering(
              n_clusters=5, linkage="ward"),
          encoded_ts, Data.y_1))


cluster_encoded = [np.argmax(x)
                   for x
                   in load_model("encoder_simple_softmax.h5").predict(Data.X_1_)]

print("Autoencoder clustering: ",
      Evaluate.evaluate_(
          cluster_encoded, Data.y_1_))

cluster_encoded_ts = [np.argmax(x)
                      for x
                      in load_model("encoder_conv_softmax.h5").predict(Data.X_1)]

print("Conv Autoencoder clustering: ",
      Evaluate.evaluate_(
          cluster_encoded_ts,
          Data.y_1))
