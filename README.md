# Restaurant Recommendation Chatbot

This work is to help me gain familiarity with prompt engineering an LLM for a specific task.

I struggle with choosing a restaurant and often forget places or meals I have had before.
This tasks is to develop a simple chatbot that can help me decide; it will have a certain persona (helpful assistant), give accurate information about my personal preferences (database of my own entries), give accurate information about other restaurants (yelp api?).

## Dataset

I am using the [yelp open dataset](https://business.yelp.com/data/resources/open-dataset/) as a teaching aid to structure my own entries to a database.

## Evaluations

### Accuracy
Given a question about a specific meal or restaurant, respond with accurate information that can be verified based on the database.

### Format
Returns responses in a structured format that can be verified via regex

### Efficiency
Time taken to generate X # of tokens

### Completeness
The repsonse includes answers to all parts of the prompt

## Retrieval Augmented Generation (RAG)

This project will likely require a vector database to store personal entries on restaurants and meals.
I think I also want to include information retrieval on other restaurants from some kind of Yelp API if that exists.
