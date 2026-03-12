# Movie_Recommendation_System
Content-based movie recommendation system using cosine similarity and TMDB movie dataset.

Movie Recommendation System

This project implements a content-based movie recommendation system using the TMDB 5000 Movie Dataset. The system analyzes movie content and recommends films that are similar to a given movie based on textual and metadata features.

The recommendation process is based on Cosine Similarity applied to vectorized movie features extracted from the dataset.

**Dataset**

The project uses the following datasets:

tmdb_5000_movies.csv

tmdb_5000_credits.csv

These datasets contain information about approximately 4800 movies, including:

movie titles

genres

keywords

cast

crew

movie overview

**Feature Engineering**

To build a meaningful recommendation system, multiple movie attributes were combined to represent the content profile of each film.

The following features were used:

Feature	Description
Overview	Short description of the movie
Genres	Movie categories (Action, Comedy, Sci-Fi, etc.)
Keywords	Important topics or themes related to the movie
Cast	Main actors appearing in the movie
Director	Director of the movie extracted from the crew column

These features were merged into a single text representation called tags.

Example representation:

action adventure alien future samworthington jamescameron

This combined representation allows the system to capture both story content and stylistic elements of a movie.

**Methodology**

The recommendation pipeline follows these steps:

Load movie and credits datasets

Merge datasets using movie title

Select relevant columns

Handle missing values

Extract important information from JSON-like columns

Combine features into a single tags column

Convert text data into numerical vectors using CountVectorizer

Compute movie similarity using Cosine Similarity

Recommend the most similar movies


**Recommendation Algorithm**

The similarity between movies is calculated using Cosine Similarity.

Movies with the highest similarity scores are recommended to the user.

Example:

recommend("Avatar")

Example output:

John Carter
Guardians of the Galaxy
Star Trek
Interstellar
The Fifth Element
