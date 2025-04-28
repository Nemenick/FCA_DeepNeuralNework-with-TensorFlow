Our aim is to learn when a Ising2D configuration is above the critical temperature!

Files: 	
Ising2d.py			-	equilibrates a 2D ising spin configuration with BVK periodic boundary conditions
conf_file_generator.py		-	generates some configurations (writing on files)
deepIsing_handmade.py	-	is the Python implementation of the network with one hidden layer (containing two hidden neurons; You alredy saw this part in the course, implemented  in fortran, by prof. Corberi)
Deep_Dense_ising.py		- 	build a Neural network made up of only FullyConnected layers, called Dense layers. (this is conceptually the same of deepIsing_basic_momentum.py, but you now are using Tensorflow!)
Deep_Conv_ising.py		-	build a Convolutional Neural network made up of FullyConnected layers and/or Convolutional layers

files *.dat are our data. You can generate new data by yourself using conf_file_generator.py (it can be very slow). Some data are alredy provided attached.


To install the needed libraries, run the commands contained in "Run_this_for_installing.ipynb". Follow the steps:
-	select your virtual environment in vscode, or pycharm (or don't activate if you want to install the libraries globally);
-	Run the commands in each cell.
