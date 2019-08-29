Face Recognition using Python
------------------------------------
Author: Fathima Nazimudeen

Requirements:
1) Python 3.5.
   numpy 1.17
   Pillow 6.1
   pip 19.2.1
   scipy 1.3
2) ORL Database (can download from the site: https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)

How to run:
    Run the file facerecognition.py. 
	The application performs training stage and then a dialog box appears asking for input. 
	Enter an input number between 1 and 12 indicating one of the 12 persons.
	If the input that you entered is between 1 and 10, then output should be an "Authenticated" message and the name of the matched person.
	If the input that you entered is between 11 or 12, the output should be "Not Authenticated" message.

Guide:
    The initial stage is the training process in which the facial features (Eigenfaces) are extracted from a set of 100 images which represents 10 different persons with 10 distinct images for each person.
    These training images are then projected on to a low dimensional feature space is using the extracted Eigenfaces.
	
	Now for the testing stage, we use TestDatabse with 12 images which contain one image each of the 10 persons used in the training and additional 2 images of the persons that were not used in the training phase. 
	This extra addition is made to find the false recognition rate.
	
    A user input image is taken from the TestDatbase and this image is also then projected on to a low dimensional feature space using the Eigenfaces extracted in the training phase. 
	Then the Euclidian Distance Classifier is used to compare the new image with each of the 100 images from the training database to find the match.
	The matched image is the one with minimum Euclidian Distance with a maximum threshold of 0.3. 
	This threshold was chosen using trial and error method and the image is rejected if the threshold is greater than 0.3.
	
	Finally, if a match is found, an "Authenticated" message and the name of the matched person will be displayed.
	And if a match is not found, a "Not Authenticated" message will be displayed.



Please contact me for any more information at
fathimanazimudeen@gmail.com