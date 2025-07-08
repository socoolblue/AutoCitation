# Retrieval-Augmented Auto Citation Suggestion

**Enhancing Academic Writing through Context-Aware Reference Selection**

## Overview

**Retrieval-Augmented Auto Citation Suggestion** is a citation recommendation system designed to assist researchers and writers in accurately citing relevant prior work. By combining large language models (LLMs) with retrieval-augmented generation (RAG), our methods improves citation reliability in academic writing by leveraging contextual document retrieval.

## Motivation

Citing appropriate references is a time-consuming and error-prone process. Existing LLMs often hallucinate or suggest irrelevant or non-existent papers. This project addresses that problem by integrating retrieval into the generation process, increasing both the relevance and accuracy of citations.

## Key Features

* Context-aware citation suggestion using RAG
* Reference repository integration for precise retrieval
* Reduced hallucinations compared to vanilla LLM prompting
* Scalable and modular architecture for extension to various academic fields

## How to use

Open Rag\_embedding\_gemini.ipynb with jupyter notebook and assign the folder path for the directory containing the reference documents to the txt\_folder variable.
Import Rag\_embedding\_gemini.ipynb.

After Rag\_embedding\_gemini.ipynb is complete, run Rag\_matching\_gemini.ipynb to check the accuracy of reference matching.

* In the intro\_text variable, insert the introduction section of the 'main paper', which serves as the basis for comparison.
* When you run the Rag\_matching\_gemini.ipynb code, it finds the most similar reference documents to the input introduction and outputs the results along with the accuracy (%).

## Reference

Seo, Y.H., Lee, B.D., Oh, Y., Hong, J.C., Moon, S.M., Jang, J.Y., Kim, M.S., Kim, J.S., \& Sohn, K.-S.  
*Retrieval-Augmented Citation Suggestion: Enhancing Academic Writing through Context-Aware Reference Selection*  
Sejong University \& Kyungpook National University

## Contact

Kee-Sun Sohn — kssohn@sejong.ac.kr

## License

This project is licensed under the MIT License.

