# Importing the cv2 module
import cv2
# Importing the decode function from pyzbar module
from pyzbar.pyzbar import decode

# The image file of barcode or QR that we want to extract text from
imgOfBarcodeOrQR = cv2.imread('qr2.png')

# Loop through the codes in the image
for code in decode(imgOfBarcodeOrQR):
    # Print the type of code (barcode or QR)
    print(code.type)
    # Print the data of the code, which is the text encoded in the barcode or QR
    print(code.data.decode('utf-8'))

# Create a text file and write the extracted text to it
textFile = open("Barcode-QR-text.txt","w")
textFile.write(str((code.data.decode('utf-8'))))
textFile.close()
