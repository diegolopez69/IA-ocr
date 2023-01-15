from PIL import Image
import pytesseract as pt

img_file = 'imagen10.png'
print ('Opening Sample file using Pillow')
img_obj = Image.open(img_file)
print ('Converting %s to string'%img_file)
ret = pt.image_to_string(img_obj)
print ('Result is: ', ret)

textFile = open("textoDetectadoDeLasImagenes.txt","w")
textFile.write(str((ret)))
textFile.close()