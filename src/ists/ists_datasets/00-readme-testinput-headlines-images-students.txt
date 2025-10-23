
	  	             STS 2016 Task 2

               Interpretable Semantic Textual Similarity


			    TEST DATASET
				   

This set of files describes the test DATASET for the Interpretable Semantic Textual Similarity task.

The test dataset contains the following:

  00-README.txt	this file

Three datasets, answers-students, headlines and images

  STSint.testinput.headlines.sent1.txt		First sentence in input sentence pairs (headlines)
  STSint.testinput.headlines.sent2.txt		Second sentence in input sentence pairs (headlines)
  STSint.testinput.images.sent1.txt		First sentence in input sentence pairs (images)
  STSint.testinput.images.sent2.txt		Second sentence in input sentence pairs (images)
  STSint.testinput.answers-students.sent1.txt	First sentence in input sentence pairs (answers-students)
  STSint.testinput.answers-students.sent2.txt	Second sentence in input sentence pairs (answers-students)

  STSint.testinput.headlines.sent1.chunk.txt		First sentence in input sentence pairs, with gold standard chunks (headlines)
  STSint.testinput.headlines.sent2.chunk.txt		Second sentence in input sentence pairs, with gold standard chunks (headlines)
  STSint.testinput.images.sent1.chunk.txt		First sentence in input sentence pairs, with gold standard chunks (images)
  STSint.testinput.images.sent2.chunk.txt		Second sentence in input sentence pairs, with gold standard chunks (images)
  STSint.testinput.answers-students.sent1.chunk.txt	First sentence in input sentence pairs, with gold standard chunks (answers-students)
  STSint.testinput.answers-students.sent2.chunk.txt	Second sentence in input sentence pairs, with gold standard chunks (answers-students)

The gold standard for the three datasets:

  STSint.testinput.headlines.wa	Gold standard alignment for each sentence pair in input (headlines)
  STSint.testinput.images.wa	Gold standard alignment for each sentence pair in input (images)
  STSint.testinput.answers-students.wa	Gold standard alignment for each sentence pair in input (students-answers)

Scripts

  wellformed.pl	Script to check for well-formed output
  evalF1.pl	Official evaluation script


Please check http://alt.qcri.org/semeval2016/task2/index.php?id=detailed-task-description for details.

Authors
-------

Eneko Agirre
Montse Maritxalar
German Rigau
Larraitz Uria
