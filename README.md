# Beyond Strict Rules: Assessing the Effectiveness of Large Language Models for Code Smell Detection

Welcome to the README of the replication package of “Beyond Strict Rules: Assessing the Effectiveness of Large Language Models for Code Smell Detection”. In this document we will briefly describe the steps to replicate our studies.

## Overview

The replication package is organized as follows:

### Automated_Detections

It contains the data of the evaluation of both Static Analysis Tools and LLMs that led to the metrics studied (F1, Precision, Recall) for all 9 smells in the study.

### Codes

It contains the 268 codes used in the study.

### Experiment

It contains all the data generated during the execution of the empirical experiment. Note that the forms are mostly in english while the participants answers are mostly in portuguese. 
In this folder you will find:

#### Background Form

The background form in which we asked the participants of the study some prelinary questions such as their experience detecting code smells, with the Java language and others.
In this folder you will also find the completely anonymized anwsers given by the participants, provinding demographic information about them.

#### Code Smell Form Answers

As described in the paper, we asked the participants of the experiment to answer some questions regarding the presence of a smell in a code.
In this folder you will find an example of such form and the completely anonymized anwsers given by the participants for each code smell.
The participants were allowed to answer the forms in english or portuguese.

#### Allocation Sample

Each participant of the experiment was asked to review between 10 to 15 codes in order to find or not the presence of a code smell.
This file presents an example of the allocation table each participant received during the experiment.

#### Consent Form

Every single one of the 78 participants of the experiment consent with the experiment by filling this form.

#### Oracle

This file presents the results of the voting for each smell which led to the creation of the 9 ground truths (one for each smell) of the study.
More details on the creation of the oracle and the ground truth are present in the paper.

#### Preparatory Lecture

Prior to the experiment, we presented the students with an 1-hour lecture allowing them to get familiarity with the smells covered in the experiment. 
This file contains the slides we used during this lecture.

### LLM

This folder contains the prompts sent to the LLMs and the outputs produced by them.  

#### How to collect the results from the LLMs

Note (1): You need to have python installed on your machine to run the scripts.
Note (2): Both scripts can be changed to whateaver HuggingFace or OpenAI model you desire. All you need is to change the model in the variable located inside the script.

1. Download and unpack the folder of with the codes;
2. Go the the "LLMs/Prompts" folder and open the script of your prompt of choice in a code editor;
3. Add the path to the folder with the codes you downloaded and umpacked earlier;
4. Add your HuggingFace or OpenAI token where is described in the script;
5. Change the destination folder in the script;
6. Open a terminal with the path set where the folder with the codes is located;
7. Run the script with python3 "SCRIPT_NAME" (MacOS) or python hello.py (Windows);

### Static Analysis Tools Results

This folder contains the results of the analysis produced by the four static analysis tools in this study for each smell for all 30 systems.

