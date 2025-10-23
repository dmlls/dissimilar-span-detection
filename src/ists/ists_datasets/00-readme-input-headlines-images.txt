
	  	             SEMEVAL 2016 Task 2

               Interpretable Semantic Textual Similarity


			    TRAIN DATASET
				   

This set of files describes the train DATASET for the Interpretable Semantic Textual Similarity task.

The train dataset contains the following:

  00-README.txt 		         this file

Two datasets, headlines and images

  STSint.input.headlines.sent1.txt       First sentence in input sentence pairs (headlines)
  STSint.input.headlines.sent2.txt       Second sentence in input sentence pairs (headlines)
  STSint.input.images.sent1.txt          First sentence in input sentence pairs (images)
  STSint.input.images.sent2.txt          Second sentence in input sentence pairs (images)

  STSint.input.headlines.sent1.chunk.txt First sentence in input sentence pairs, with gold standard chunks (headlines)
  STSint.input.headlines.sent2.chunk.txt Second sentence in input sentence pairs, with gold standard chunks (headlines)
  STSint.input.images.sent1.chunk.txt    First sentence in input sentence pairs, with gold standard chunks (images)
  STSint.input.images.sent2.chunk.txt    Second sentence in input sentence pairs, with gold standard chunks (images)

  STSint.input.headlines.wa		 Gold standard alignment for each sentence pair in input (headlines)
  STSint.input.images.wa			 Gold standard alignment for each sentence pair in input (images)

Scripts

  wellformed.pl                          Script to check for well-formed output
  evalF1.pl                              Official evaluation script



Train Data Description
-----------------------

The train data comprises the iSTS 2015 train and test pairs, which have
been re-annotated according to the 2016 guidelines (see "Updates with
respect to SemEval-2015" section of the guidelines for differences,
which concern mostly the ALIC case).

The dataset was sampled from previous STS datasets:

- headlines: Headlines mined from several news sources by European
  Media Monitor using their RSS feed.
  http://emm.newsexplorer.eu/NewsExplorer/home/en/latest.html

- images: The Image Descriptions data set is a subset of the Flickr
  dataset presented in (Rashtchian et al., 2010), which consisted on
  8108 hand-selected images from Flickr, depicting actions and events
  of people or animals, with five captions per image. The image
  captions of the data set are released under a CreativeCommons
  Attribution-ShareAlike license.



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

