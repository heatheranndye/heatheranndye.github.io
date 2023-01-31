title: Temperature Quilts in Python
author: Heather Ann Dye
date: 1/31/2023
category: data science, art 
tags: pandas, pillow, streamlit

## What is a temperature quilt? 

A temperature quilt displays the daily temperature data from a specific date range. Fixed temperature ranges correspond to specific colors - so
the quilt (or image) is a visual record of the climate.  I constructed this app as a demonstration project for data storytelling. In future projects, I'll be using Pillow and matplotlib along with pandas and sci-kit learn to construct stories about data and possibly transcribing them into works of art.
I've got some unique constraints and requirements for the project (including a pdf quilt pattern) since the goal of the project is to autom
#### Example Quilts and the Streamlit App

The streamlit app is located at [DataQuilt](https://h-a-dye-dataquilt-streamstreamlit-app-zwncqy.streamlit.app/ ). Once you've opened the app, simply put in your US zip code and year.  Then the app will automatically search for the closest weather station that contains a maximum amount of data for that year.
The information comes from the National Oceanic and Atmospheric Administration's (NOAA) Global Historical Climatology Network daily (GHCNd) of weather stations. Once the closest weather station with the most data is located (not all weather stations have a complete data set), the data is binned and each bin is associated with a particular color. From this, we can automatically generate a diagram of the temperatures and a corresponding pattern for the physical construction of the quilt. 

### The backend of the app

#### Pandas and requests and 

NOAA maintains an inventory of 