# Language_Identification

Language identification system based on InceptionV3 and InceptionV3-CRNN (implemented using Tensorflow and Keras). The repository contains importing scripts for CommonVoice free datasets.
globalparam.py file contains global parameters for downloading, importing, training, evaluating, and testing processes.
The spectrogram generation process uses Nyquist–Shannon sampling theorem. Each spectrogram by default contains a maximum of 50 pixels(this is also configurable).

### Environment and Requirements
  * OS: Linux Ubuntu 20.04.4 LTS
  * Python 3 version: 3.8.10.
  * Pip 3 version: 21.3.1.

Supported classification models are:
  - [x] InceptionV3
  - [x] Inceptionv3 + CRNN



Install the required dependencies:
> pip3 install -r requirements.txt
