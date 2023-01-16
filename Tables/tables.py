# Import the necessary libraries
from io import StringIO
import cv2
import pytesseract
import pandas as pd
import numpy as np

# Read an image file and store it as a variable
image = cv2.imread('table.png')

# Use pytesseract to extract the text from the image
text = pytesseract.image_to_string(image)

# Print the extracted text
print(text)

# Split the text into lines
lines = text.splitlines()

# Create an empty list to store the data
data = []

# Iterate over the lines of text
for line in lines:
    # Split the line into fields
    fields = line.split(" ")
    # Remove leading and trailing whitespace from each field
    fields = [field.strip() for field in fields]
    # Add the fields to the data list
    data.append(fields)

# Print the data
print(data)

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Remove any columns that are entirely empty
df = df.dropna(axis=1, how="all")

# Convert the DataFrame to an HTML table
html_table = df.to_html(index=False, header=False)

# Print the HTML table
print(html_table)

# Create a text file to write the HTML table
textFile = open("CodigoDetectadoParaGenerarTablas.txt","w")

# Write the HTML table to the text file
textFile.write(str((html_table)))

# Close the text file
textFile.close()
