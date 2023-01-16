# Importing 
from PIL import Image
import pytesseract as pt

# The image file that we want to extract text from
imgWithTextToConvert = 'digitalToText2.png'

# Open the image file using the Image module
img_obj = Image.open(imgWithTextToConvert)

# Print message to show that we are converting the image to text
messageToSend = 'Converting %s to string \n\n'%imgWithTextToConvert 
print (messageToSend)

# Use the image_to_string function from the pytesseract module to extract text from the image
digitalToText = pt.image_to_string(img_obj)

# Print the text that was extracted
print ('Result is: ', digitalToText)

# Create a text file and write the extracted text to it
textFile = open("imageToText.txt","w")
textFile.write(str((messageToSend)))
textFile.write(str((digitalToText)))
textFile.close()
