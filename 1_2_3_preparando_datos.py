from tqdm import tqdm
import json
from simhash import Simhash, SimhashIndex
from collections import Counter
import random
import spacy
from nltk.tokenize import TextTilingTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import pickle

# Cargar el modelo de lenguaje de spaCy
nlp = spacy.load("es_core_news_sm")

# Cargar las stopwords en español
stopwords_spanish = set(stopwords.words('spanish'))

# Ruta del archivo de noticias
ruta_archivo = "noticias-cambio-climático-españa-abril-2022-abril-2023-finales-sentencias-disagree-NOT-disagree.ndjson"

#eliminar cuasi-duplicados
 #carga de las noticias en memoria
lineas = open(ruta_archivo, "r").readlines()

textos = list()

with tqdm(total=len(lineas)) as barra:
    for linea in lineas:
        try:
            data = json.loads(linea)
            if len(data["sentencias_disagree"]) < len(data["sentencias_NOT_disagree"]):
                texto = " ".join(data["sentencias"])
                texto = "".join(texto.splitlines())
                textos.append(texto)
        except:
            pass

        barra.update(1)

valor_f = 128



firmas = []
with tqdm(total=len(textos)) as barra:
    for i in range(len(textos)):
        texto = textos[i]

        firma = Simhash(texto, f=valor_f)
        firmas.append((i, firma))

        barra.update(1)
observaciones = []
indice = SimhashIndex(firmas, k=10, f=valor_f)

with tqdm(total=len(textos)) as barra:
    for i in range(len(textos)):
        firma = firmas[i][1]
        duplicados = indice.get_near_dups(firma)

        observaciones.append(len(duplicados))

        barra.update(1)

print()
print(Counter(observaciones).most_common(50))
textos_no_duplicados = list()

ignorar = list()

with tqdm(total=len(textos)) as barra:
    for i in range(len(textos)):
        if i not in ignorar:
            texto = textos[i]

            firma = firmas[i][1]

            duplicados = indice.get_near_dups(firma)

            if len(duplicados) == 1:
                textos_no_duplicados.append(texto)
                ignorar.append(i)
            else:
                random.shuffle(duplicados)

                ident = int(duplicados[0])

                textos_no_duplicados.append(textos[ident])
                for ident in duplicados:
                    ignorar.append(int(ident))

        barra.update(1)

print()
print(len(textos_no_duplicados))


#segmentando con spacy 
textos_segmentados = []

with tqdm(total=len(textos_no_duplicados)) as barra:
  for doc in textos_no_duplicados:
    sentencias = nlp(doc).sents
    textos_segmentados.append("\n\n".join(map(str, sentencias)))
    barra.update(1)
  
  tt = TextTilingTokenizer(stopwords=set(stopwords.words('spanish')))
lista_segmentos = []


#textiling
with tqdm(total=len(textos_segmentados)) as barra:
  for doc in textos_segmentados:
    try:
      lista_segmentos += tt.tokenize(doc) 
    except:
      pass
    barra.update(1)

# Imprimir los segmentos
for i, segmento in enumerate(lista_segmentos):
    print(i, segmento.strip())
    print("---")

# Clustering de segmentos


vectorizador = TfidfVectorizer(encoding="utf-8", lowercase=True, stop_words=list(spacy.lang.es.stop_words.STOP_WORDS), ngram_range=(1, 3), max_features=10000)
doc_term_matrix = vectorizador.fit_transform(lista_segmentos)

num_clusters = 50
clustering1 = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=1000, n_init=1, verbose=True)
clustering1.fit(doc_term_matrix)

clustered_docs1 = dict()
docs_per_cluster1 = dict()

for i in range(len(clustering1.labels_)):
    try:
        clustered_docs1[clustering1.labels_[i]].append(i)
        docs_per_cluster1[clustering1.labels_[i]] += 1
    except:
        clustered_docs1[clustering1.labels_[i]] = list()
        clustered_docs1[clustering1.labels_[i]].append(i)
        docs_per_cluster1[clustering1.labels_[i]] = 1

ids = list(docs_per_cluster1.keys())
ids.sort()
sorted_docs_per_cluster1 = {i: docs_per_cluster1[i] for i in ids}

terminos = vectorizador.get_feature_names_out()
indice_cluster_terminos1 = clustering1.cluster_centers_.argsort()[:, ::-1]
clustered_docs_with_segments1 = dict()

for cluster_id in clustered_docs1:
    segmentos_cluster = [lista_segmentos[i] for i in clustered_docs1[cluster_id]]
    clustered_docs_with_segments1[cluster_id] = segmentos_cluster

# Guardar el resultado del clustering en un archivo utilizando pickle
with open("resultado_clustering.pkl", "wb") as archivo_clustering:
    pickle.dump(clustered_docs_with_segments1, archivo_clustering)

    

print("Resultados del clustering guardados en el archivo resultado_clustering.pkl.")

for cluster_id in sorted_docs_per_cluster1:
    print("Cluster %d (%d documentos): " % (cluster_id, docs_per_cluster1[cluster_id]), end="")

    for term_id in indice_cluster_terminos1[cluster_id, :10]:
        if clustering1.cluster_centers_[cluster_id][term_id] != 0:
            print('"%s"' % terminos[term_id], end=" ")

    print()

    ejemplares = clustered_docs1[cluster_id]
    random.shuffle(ejemplares)
    for ejemplar in ejemplares[0:5]:
        print("\t", lista_segmentos[ejemplar][0:140], "...")

    print()

