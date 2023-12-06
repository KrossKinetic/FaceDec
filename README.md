# WindowsCenterStage 

A simple python program using TensorFlow image recognition model to emulate iPad's CenterStage on windows.

Huge thanks to Nicholas Renotte for his YouTube video which helped me learn how to make an image detection model.

This is just a proof of concept. I couldn't find a simple and non-resource intensive way to turn the opencv frame output into a virtual webcam for apps like Zoom to access. That is something I am still working on. This works fine on my machine but might not work on yours. Make sure to check the width and height of the camera you are using and replace the values whereever I have put a '#!'. Experiment and find the value that is suitable for you.

Since I am just a college student, the model I have made was trained on approximately 6000 images. The model is pretty good but can be improved by training it on more than 100,000 images, something which I do not have time or resources for. 

I have also now added the Jupyter Notebook Python code for the Face Recognition Model. Big thanks to Nicholas once again for teaching me how to code that.