from datetime import datetime
from elasticsearch import Elasticsearch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
# Generamos el objeto cliente que nos permite establecer la conexión con Elasticsearch
es = Elasticsearch ('http://192.168.1.24:9200')
# Obtención de un documento de Elasticsearch a partir de su indice e id del documento
resp = es.get(index='metricbeat-7.13.2-2021.06.18-000001',id='')
# Leemos 10.000 documentos del indice metricbeat-*
docs = es.search (index='metricbeat-7.13.2-2021.06.18-000001')
hits =docs['hits']['hits']
# Para este ejercicio en concreto nos interesa recopilar memoria y cpu
metrics = {'memory': [], 'cpu':[]}
process = []
# The percentage of CPU time spent by the process since the last update.
# Its value is similar to the %CPU value of the process displayed by the top command on Unix systems.
#
# The percentage of memory the process occupied in main memory (RAM).
for hit in hits:
    if hit['_source']['metricset']['name']=='process':
        metrics['memory'].append(hit['_source']['system']['process']['memory']['res']['pct'])
        metrics['cpu'].append(hit['_source']['system']['process']['cpu']['total']['pct'])
        process.append[hit]
# Visualización de los datos extraidos
leng(metrics['memory'])
# Utilizamos la libreria pandas para visualizar los datos
df = pd.DataFrame(metrics)
# Visualización de los 10 primeros valores
df.head(10)
df.describe()
# Representación gráfica de los valores de la memoria y cpu
plt.figure(figsize=(14, 6))
plt.scatter(df["memory"], df["cpu"], c="b", marker=".")
plt.xlabel("memory", fontsize=14)
plt.ylabel("cpu", fontsize=14)
plt.show()
ift_clf = IsolationForest(contamination=0.002, max_samples=300)
ift_clf.fit(df)


# Representación gráfica del límite de decisión generado
def plot_isolation_forest(X, resolution=1000):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))

    Z = ift_clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z)

    plt.plot(X[:, 0], X[:, 1], 'w.')


plt.figure(figsize=(14, 6))
plot_isolation_forest(df.values)
plt.xlabel("memory", fontsize=14)
plt.ylabel("cpu", fontsize=14)
plt.show()
# Identificación de anomalías
anomalies = ift_clf.predict(df)
print("Total de anomalías identificadas:", len(df[anomalies==-1]))
# Representación gráfica de las anomalías
plt.figure(figsize=(14, 6))
plt.plot(df["memory"][anomalies == -1],df["cpu"][anomalies == -1], 'go', markersize=6)
plot_isolation_forest(df.values)
plt.xlabel("memory", fontsize=14)
plt.ylabel("memory", fontsize=14)
plt.show()
df[anomalies==-1]
process[1075]