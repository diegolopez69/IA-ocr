from io import StringIO
import cv2
import pytesseract
import pandas as pd
import numpy as np

# Read the image and convert it to grayscale
image = cv2.imread('table3.png')
# Extraer el texto de la imagen utilizando el modo de rendimiento 1
text = pytesseract.image_to_string(image)


# Extract the text from the image
#text = pytesseract.image_to_string(brightness)
print(text)
# Dividir el texto en líneas
lines = text.splitlines()

# Crear una lista de listas con los campos de cada línea
data = []
for line in lines:
    fields = line.split(" ")
    fields = [field.strip() for field in fields]
    data.append(fields)

print(data)
# Crear un DataFrame de Pandas a partir de la lista de listas
df = pd.DataFrame(data)

# Eliminar las columnas vacías del DataFrame
df = df.dropna(axis=1, how="all")

# Generar la tabla HTML
html_table = df.to_html(index=False, header=False)

# Imprimir la tabla HTML
print(html_table)

# Poner el código en notas
textFile = open("CodigoDetectadoParaGenerarTablas.txt","w")
textFile.write(str((html_table)))
textFile.close()