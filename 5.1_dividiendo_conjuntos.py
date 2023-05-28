import random

# Cargar los segmentos etiquetados desde el archivo
with open("clusters_etiquetados.txt", "r") as archivo_etiquetas:
    lineas = archivo_etiquetas.readlines()

# Eliminar líneas en blanco y saltos de línea adicionales
lineas = [linea.strip() for linea in lineas if linea.strip()]

# Realizar shuffle en la lista de segmentos
random.shuffle(lineas)
print("Cantidad de lineas:", len(lineas))
# Dividir en conjunto de entrenamiento y prueba
porcentaje_entrenamiento = 0.8  # Porcentaje de segmentos para entrenamiento
num_segmentos_entrenamiento = int(len(lineas) * porcentaje_entrenamiento)

segmentos_entrenamiento = lineas[:num_segmentos_entrenamiento]
segmentos_prueba = lineas[num_segmentos_entrenamiento:]

# Imprimir la cantidad de segmentos en cada conjunto
print("Cantidad de segmentos en el conjunto de entrenamiento:", len(segmentos_entrenamiento))
print("Cantidad de segmentos en el conjunto de prueba:", len(segmentos_prueba))

# Guardar los segmentos de entrenamiento en un archivo
with open("segmentos_entrenamiento.txt", "w") as archivo_entrenamiento:
    archivo_entrenamiento.write("\n".join(segmentos_entrenamiento))

# Guardar los segmentos de prueba en un archivo
with open("segmentos_prueba.txt", "w") as archivo_prueba:
    archivo_prueba.write("\n".join(segmentos_prueba))
