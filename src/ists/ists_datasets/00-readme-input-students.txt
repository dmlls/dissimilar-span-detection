
	  	             SEMEVAL 2016 Task 2

               Interpretable Semantic Textual Similarity


			    STUDENT ANSWERS DATASET
				   

This set of files describes the student answers train DATASET for the Interpretable Semantic Textual Similarity task.

The student answers train dataset contains the following:

  00-README.txt 		         this file

One dataset, students answers

  STSint.input.answers-students.sent1.txt	First sentence in input sentence pairs 
  STSint.input.answers-students.sent2.txt	Second sentence in input sentence pairs

  STSint.input.answers-students.sent1.chunk.txt	First sentence in input sentence pairs, with gold standard chunks
  STSint.input.answers-students.sent2.chunk.txt	Second sentence in input sentence pairs, with gold standard chunks

  STSint.input.answers-students.wa	Gold standard alignment for each sentence pair in input



Train Data Description
-----------------------

The student answer corpus consists of the interactions between students and the BEETLE II tutorial dialogue system.
The BEETLE II system is an intelligent tutoring engine that teaches students in basic electricity and electronics.
At first, students spend three to five hours reading material, building and observing circuits in the simulator and interacting with a dialogue-based tutor.
They used the keyboard to interact with the system, and the computer tutor asked them questions and provided feedback via a text-based chat interface.
The data from 73 undergraduate volunteer participants at south-eastern US university were recorded and annotated to form the BEETLE human-computer dialogue corpus.

In the present corpus, we include sentence pairs composed of an student answer and the reference answer of a teacher.
We have rejected those answers containing pronouns whose antecedent is not in the sentence (pronominal coreference),
because, as the question is not included in the train data, we can not deduce which is the antecedent.

There are some aspects which are specific to this corpus and have to be taken into account:
  - A, B and C refer to bulb A, B and C. 
  - X, Y, and Z refer to switches X, Y, and Z.
  - When numbers appear alone in a chunk, they refer to circuits.      
  - By default there is a unique battery, unless it is not explicitly mentioned.
  - By default paths are considered to be closed.


For more information refer to: 

Dzikovska et al (2010). Intelligent tutoring with natural language support in the Beetle II system.
In Sustaining TEL: From Innovation to Learning and Practice (pp. 620-625). Springer Berlin Heidelberg.

or

Dzikovska et al (2012, June). Towards effective tutorial feedback for explanation questions: A dataset and baselines.
In Proceedings of the 2012 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (pp. 200-210). Association for Computational Linguistics.



Please check http://alt.qcri.org/semeval2016/task2/index.php?id=detailed-task-description for details on:

- Task introduction
- General description
- Input format
- Gold Standard annotation format
- Answer format
- Scoring


Authors
-------

Eneko Agirre
Aitor Gonzalez-Agirre
Inigo Lopez-Gazpio
Montse Maritxalar
German Rigau
Larraitz Uria

