import cv2
from pyzbar.pyzbar import decode

img = cv2.imread('qr2.png')

for code in decode(img):
    print(code.type)
    print(code.data.decode('utf-8'))

textFile = open("textoDetectadoDeCodigosDeBarras_Qr.txt","w")
textFile.write(str((code.data.decode('utf-8'))))
textFile.close()