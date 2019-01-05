
An implementation of an Encoder:

To have a better idea about the project, you can follow the instructions below:

• Go through the tutorial “Classifying Names with a Character-Level RNN”
http://pytorch.org/tutorials/intermediate/char rnn classification tutorial.html
• Download the data associated with the tutorial. In a Python notebook within Google Colab(if you are using one), use the following
instructions to download the data into the working directory of the virtual machine:
!wget https://download.pytorch.org/tutorial/data.zip
!unzip data.zip
• Run the script at the end of the tutorial

The project compares the accuracy of an encoder when the type of hidden units(specifically the linear units, long short term memory (LSTM) units and gated reccurent units(GRU)) are varied. The coparisons are documents in the results.docx file.
