# segvoice

Segvoice was originally used to extract user voices from customer service calls. 
Because after that we need perform speaker verification on massive user voices, this tool tries to ensure the acquired speaker voices is pure. If you have similar needs, then it also works for you.

####  How to get pure speaker voices
* use large training data
* only accept low score segments
* remove silent segments 
* merge segments if they belong to one speaker, and only keep long merge segments 
* thanks to many awesome tools, python_speech_features for extract mfcc feature, 
scikit_learn for train GMM models, numpy for matrix computing

#### How to use it 
* install python dependencies => pip install -r requirements 
* install local tool => sox  

####  Experiment
I choose some speech from thchs30 and random cat small segments to simulate calls.
![](https://github.com/lianghyv/segvoice/demo.jpg)