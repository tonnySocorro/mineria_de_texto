import json
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize.texttiling import TextTilingTokenizer
from tqdm import tqdm
from nltk.tokenize import TextTilingTokenizer
from nltk.corpus import stopwords

# Ruta del archivo de noticias
ruta_archivo = "noticias-cambio-climático-españa-abril-2022-abril-2023-finales-sentencias-disagree-NOT-disagree.ndjson"

lineas = open(ruta_archivo, "r").readlines()

noticias_negacionistas = list()

with tqdm(total=len(lineas)) as barra:
    for linea in lineas:
        try:
            data = json.loads(linea)
            if len(data["sentencias_disagree"]) > len(data["sentencias_NOT_disagree"]):
                texto = " ".join(data["sentencias"])
                texto = "".join(texto.splitlines())
                noticias_negacionistas.append(texto)
        except:
            pass

        barra.update(1)

valor_f = 128
segmentos = []

# Segmentación de las noticias no negacionistas en sentencias y segmentos temáticamente coherentes
nlp = spacy.load("es_core_news_sm")
stopwords_spanish = set(stopwords.words('spanish'))

tt = TextTilingTokenizer(stopwords=stopwords_spanish)

textos_segmentados = []


with open("noticias_negacionistas.txt", "w") as file:
    for noticia in noticias_negacionistas:
        sentencias = nlp(noticia).sents
        textos_segmentados.append("\n\n".join(map(str, sentencias)))
        barra.update(1)

    for doc in textos_segmentados:
        lista_segmentos = []
        try:
            lista_segmentos += tt.tokenize(doc)
            segmentos.append(lista_segmentos)
            file.write("\n".join(lista_segmentos) + "\n")  # Escribir los segmentos en el archivo
            file.write("=====\n")  # Insertar separador entre noticias
        except:
            pass
        barra.update(1)
