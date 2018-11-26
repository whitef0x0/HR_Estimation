# Requirements
- numpy, opencv, dlib, imutils, scipy, PyQt4, sklearn
- Download and put file "shape_predictor_68_face_landmarks.dat" in the project folder

# Installation
** Note: This guide assumes you are using Mac OSX **

This guide assumes you have homebrew and python3 installed

1. Install dlib
	2.1 Install dlib dependencies
		```
		brew cask install xquartz
		brew install gtk+3 boost
		brew install boost-python --with-python3
		```
	2.2 Install dlib 
		```
		brew install dlib
		```
2. Setup virtual environment
	2.1 Install virtualenv
		```
		pip install virtualenv
		```
	2.2 Create virtual envirnoment
		```
		virtualenv -p python3 ./ENV
		```
	2.3 Activate virtual environment
		```
		source ENV/bin/activate
		```
3. Install pip modules
`pip3 install -r requirements.txt`
4. Install PyQt4
`python -m pip install PyQt4`
5. Start program
`python GUI.py`
	- In case of plotting graphs, run "graph_plot.py" 
	- For the Eulerian Video Magnification implementation, run "amplify_color.py"

# Quickstart

### Running PPG method
```python3 run_method1_ppg.py /path/to/video /path/to/groundtruth```

### Running EVM
```python run_method1_evm.py /path/to/video /path/to/groundtruth```
