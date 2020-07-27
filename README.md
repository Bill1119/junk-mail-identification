# junk-mail-identification
A Machine learning-based application for junk mail identification. Finding the possible key words that related to junk mails and identifing the mail is junk mail or not.

The project consists of four (4) classes:
 <li>CountVectorizer : Responsible for computing the vector of frequency for each document</li>
 <li>Main : Contains the main method, where program execution begin.</li>
 <li>NaiveBayes : Responsible for training and prediction</li>
 <li>PreProcess : Responsible for processing the texts. Folds all characters to lowercase, then tokenize them using 
 regular expression and uses the set of resulting words as the vocabulary.</li>
 
 ### How to run:
 This program was written using Python 3.7
 <li>Dependencies: Numpy, re</li>
 <br>
 After cloning/downloading, be sure that the training and test samples are available in ../datasets/train/ and 
 ../datasets/test folders respectively.
 
 Start execution from the Main class. After completion the following files are generated:
 <li>model.txt file with the information
   <ol>
    <li>A line counter i, followed by 2 spaces.</li>
    <li>The word w<sub>i</sub>, followed by 2 spaces.</li>
    <li>The frequency of w<sub>i</sub> in the class ham, followed by 2 spaces.</li>
    <li>The smoothed conditional probability of w<sub>i</sub> in the class ham −P(w<sub>i</sub>|ham), followed by 2 spaces.</li>
    <li>The frequency of w<sub>i</sub> in the class spam, followed by 2 spaces</li>
    <li>The smoothed conditional probability of w<sub>i</sub> in the class spam −P(w<sub>i</sub>|spam), followed by a carriage return.</li>
   </ol>
  is generated and saved in the output_files folder located in the root folder.</li>
 <li>result.txt file with the information 
   <ol>
    <li>A line counter i, followed by 2 spaces.</li>
    <li>the name of the test file, followed by 2 spaces</li>
    <li>the classification as given by your classifier (the label spam or ham), followed by 2 spaces</li>
    <li>the score of the class ham as given by your classifier, followed by 2 spaces</li>
    <li>the score of the class spam as given by your classifier, followed by 2 spaces</li>
    <li>the correct classification of the file, followed by 2 spaces</li>
    <li>the label right or wrong (depending on the case), followed by a carriage return</li>
   </ol>
 is generated and saved in the output_files folder located in the root folder.</li>
 
 Accuracy, precision, recall and f1-measure are computed for spam and ham classes.
 Accuracy, precision, recall and f1-measure are computed for the overall system.
