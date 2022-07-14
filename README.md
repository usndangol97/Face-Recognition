# Face-Recognition

## Description
Implemented Face Recognition system by using pretrained faceEvolve model and Faiss Indexer. It uses opencv libraries to manipulate images, capture images for further preprocessng.Pretrained faceEvolve model is in pytorch format which we use to calculate facial embedding vectors. The facial embedding vectors is used to calculate similarity in images through which we recognize the faces in the dataset. This system is implemented from GitHub repo [faceEvolve](https://github.com/ZhaoJ9014/face.evoLVe). You can find the pretrained model of faceEvolve in [faceEvolve](https://github.com/ZhaoJ9014/face.evoLVe).

- - - - 

## Table of contents 

* [Installation](#installation)
* [Libary Usage](#usage)
* [Contributors](#contributors)
* [Tests](#tests)
* [Questions](#questions)

<a name="installation"></a>
## Installation 
Clone this github repository and open the file in any IDE of your choice. Install necessary libraries such as pytorch, tensorflow, opencv. Ensuring you are in the correct file in the terminal or directory. Make sure to change the path of your dataset in faceEvolve.py file and also make sure to specify correct path of the pretrained faceEvolve model in faceEvolve.py file. while loading the dataset in the file.

<a name="usage"></a>
## Library Usage
* openCV
* os
* numpy
* Pytorch


## Demos
```bash
# From the project root directory
Python3 faceEvolve.py
```


<a name="license"></a>
## License 
This application is released under the MIT License.


<a name="contributors"></a>
## Contributors 
* usndangol97


<a name="tests"></a>
## Tests 
No test instructions for this yet

<a name="questions"></a>
### Questions

If you have any questions, visit my [GitHub profile](https://www.github.com/usndangol97) or email me: usndangol97@gmail.com
