# Face Recognition using PCA in Python

from PIL import Image
import numpy as np
import scipy.sparse.linalg
import tkinter as tk
from tkinter import simpledialog
gama = np.zeros((307200,100))# here 100 face images from orl database is used for training
print("Opening images and representing each image as column of single matrix gama")
for x in range(1,101):
 img = Image.open("TrainDatabase/{}.jpg".format(x))
 g=img.convert("L")
 temp=np.transpose(g)
 element_Count=temp.size
 t=np.reshape(temp,(element_Count,1))
 gama[:,x-1]=t[:,0]
print("gama matrix generated")
gama = np.matrix(gama)
print("Creating average face vector")
mean_All = gama.mean(1) # returns mean value of each row as a column matrix
A = np.zeros((307200,100))
A=np.matrix(A)
print("Calculating co-variance matrix")
for x in range(0,100):
 y = gama[:,x]
 t = np.subtract(y,mean_All)
 A[:,x]=t[:,0]
W = A.T
L = W.dot(A)
print("Co-variance matrix calcuated")
print("Computing EigenFaces")
P=50 # to select p largest eigen values
Eigen_Values, Eigen_Vectors = scipy.sparse.linalg.eigs(L, k=P)
EigenFaces = A.dot(Eigen_Vectors)
ProjectedImages = np.zeros((50,100),dtype=np.complex_)
ProjectedImages=np.matrix(ProjectedImages)
E = EigenFaces.T
print("Computed EigenFaces")
print("Starting Projecting images to EigenSpace")
for i in range(0,100):
    c = A[:,i]
    temp = E.dot(c)
    ProjectedImages[:,i]=temp[:,0] #warning:Casting complex values to real discards the imaginary part
print("Completed Projecting images to EigenSpace")
print("Training complete")

# Creating dialog box to give input to the face authentication system.
# here 10 images are used for training (of 10 persons). Additional 2 images of persons not in the training data set are also added to find the false recognition rate.
ROOT = tk.Tk()
ROOT.withdraw()
I = simpledialog.askstring(title="Face Recognition System",
                                  prompt="Enter test image name(a number between 1 and 10):")
InputImage = Image.open("TestDatabase/{}.jpg".format(I))
Img = InputImage.resize((480, 640),Image.ANTIALIAS)
Img=Img.convert("L")
Img.show()
Temp=np.transpose(Img)
element_Count1=Temp.size
InImage=np.reshape(Temp,(element_Count1,1))
Difference = np.subtract(InImage,mean_All)
ProjectedTestImage = E.dot(Difference)
ED = np.zeros((1,100),dtype=np.complex_)
for i in range(0,100):
 q = ProjectedImages[:,i]
 d = np.subtract(ProjectedTestImage,q)
 dt=d.T
 d1 = dt.dot(d)
 ED[:,i] = d1
val =int(1000000000000000000)
EDnew = ED/val
minimum_ED = np.amin(EDnew)
Recognized_index = np.where(EDnew == minimum_ED)
for h in Recognized_index:
    print(h)
y = h+1
if minimum_ED > 0.3:
    print("Not Authenticated")
else:
 print("Authenticated")   
 Matched_image = print("Matched image is {}.jpg".format(y))
 div = y/10
 for x in range(1,11):
    if div <= x:
        print("Matched person is {}".format(x))
        break
