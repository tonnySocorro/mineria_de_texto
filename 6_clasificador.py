import fasttext
import json
# Rutas de los archivos de entrenamiento y prueba
archivo_entrenamiento = "segmentos_entrenamiento.txt"
archivo_prueba = "segmentos_prueba.txt"

# Entrenar el modelo
modelo = fasttext.train_supervised(input=archivo_entrenamiento, epoch=30)

# Evaluar el modelo en el conjunto de prueba
resultados = modelo.test(archivo_prueba)

# Imprimir los resultados de la evaluación
print("Número de ejemplos:", resultados)
print("Precisión:", resultados[1])

# Cargar las noticias negacionistas desde un archivo de texto
with open("noticias_negacionistas.txt", "r") as archivo_noticias:
    noticias = archivo_noticias.read().split("=====")

# Guardar los segmentos etiquetados en un archivo de texto
with open("noticias_negacionistas_clasificadas.txt", "w") as archivo_etiquetas:
    # Iterar sobre las noticias y sus segmentos
    for i, noticia in enumerate(noticias):
        segmentos = noticia.strip().split("\n")
        archivo_etiquetas.write(f"Noticia {i+1} - Segmentos:\n")
        
        etiquetas_segmentos = []
        for segmento in segmentos:
            segmento = segmento.strip()
            if segmento:
                etiqueta = modelo.predict(segmento)[0][0]
                etiquetas_segmentos.append(etiqueta)
                archivo_etiquetas.write(f"  Segmento: {segmento} - Etiqueta: {etiqueta}\n")
        
        archivo_etiquetas.write("Etiquetas:\n")
        etiquetas_frecuencia = {}
        total_segmentos = len(etiquetas_segmentos)
        for etiqueta in etiquetas_segmentos:
            etiquetas_frecuencia[etiqueta] = etiquetas_frecuencia.get(etiqueta, 0) + 1
        
        for etiqueta, frecuencia in etiquetas_frecuencia.items():
            porcentaje = frecuencia / total_segmentos * 100
            archivo_etiquetas.write(f"  Etiqueta: {etiqueta} - Porcentaje: {porcentaje:.2f}%\n")
        
        archivo_etiquetas.write("\n")
