import pickle

# Diccionario de clusters seleccionados y sus etiquetas
clusters_etiquetas = {
    12: "__label__biodiversidad",
    25: "__label__co2",
    26: "__label__cambioClimático",
    27: "__label__renovables",
    29: "__label__incendios",
    31: "__label__emisiones",
    33: "__label__renovables",
    40: "__label__olaCalor",
    48: "__label__calentamiento"

}

# Cargar el resultado del clustering desde el archivo pickle
with open("resultado_clustering.pkl", "rb") as archivo_clustering:
    clustered_docs_with_segments = pickle.load(archivo_clustering)

# Seleccionar los clusters deseados
clusters_deseados = [12, 25, 26,27, 29, 31,33,40,48]

# Crear archivo de texto para guardar los clusters etiquetados
with open("clusters_etiquetados.txt", "w") as archivo_etiquetas:
    for cluster_id in clusters_deseados:
        if cluster_id in clustered_docs_with_segments:
            segmentos_cluster = clustered_docs_with_segments[cluster_id]
            segmentos_etiquetados = [clusters_etiquetas[cluster_id] + " " + segmento.replace('\n', '') for segmento in segmentos_cluster]
            
            # Escribir los segmentos etiquetados en el archivo

            for segmento_etiquetado in segmentos_etiquetados:
                archivo_etiquetas.write(segmento_etiquetado + "\n")
            archivo_etiquetas.write("\n")
        else:
            archivo_etiquetas.write(f"No se encontró el cluster {cluster_id}.\n")
