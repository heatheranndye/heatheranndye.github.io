title: Temperature Quilts in Python
author: Heather Ann Dye
date: 1/31/2023
category: data science, art 
tags: pandas, pillow, streamlit

## What is a temperature quilt? 

A temperature quilt displays the daily temperature data from a specific date range in a specific location. Colors are assigned to specific temperature ranges so that
the quilt (or image) is a visual record of the climate.  This app is a demonstration project for data storytelling as part of my participation in the [PyBites Professional Developer Mindset Program](https://pybit.es/). 

This project uses pandas, Pillow, matplotlib along with Streamlit.io to construct a data story  and a work of art! 
This project has some unique constraints and requirements (including a pdf quilt pattern) so users can construct an actual quilt.

#### A photo of my *actual* data quilt. 
![jpg](/images/data_quilt_files/actualdataquilt.jpg)

#### Example Quilts and the Streamlit App

The streamlit app is located at [DataQuilt](https://h-a-dye-dataquilt-streamstreamlit-app-zwncqy.streamlit.app/ ). Once you've opened the app, simply put in your US zip code and year.  Then the app will automatically search for the closest weather station that contains a maximum amount of data for that year.
The information comes from the National Oceanic and Atmospheric Administration's (NOAA) Global Historical Climatology Network daily (GHCNd) of weather stations. Once the closest weather station with the most data is located (not all weather stations have a complete data set), the data is binned and each bin is associated with a particular color. From this, a digital mock-up of a temperature quilt and a corresponding pattern for the physical construction of the quilt is created automatically. 

The code is located in the github repository: [H-A-DYE/DataQuilt](https://github.com/H-A-Dye/DataQuilt). 

### Data from NOAA 

NOAA provides an inventory of weather stations at [https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt]("https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt). Each line in this text file lists a weather station, along with its lattitude and longitude and years of activity. 
This information is stripped out of the text file using string methods and transformed into a pandas dataframe. The next step is to sort the dataframe based on years of availability.  Next, we use geopy to convert the provided zip code into latitude and longitude and the distance between each weather station and the zip code is computed.  The app uses the 10 closest weather stations in the next step.

### Request the data and sort with pandas

Not all weather stations have a complete set of records - so the app examines the 10 closest weather stations and selects the weather station that has the most complete set of data. I used the requests package to request the data from NOAA and extract a json data which is transformed into a pandas data frame. The data frame is examined for missing values and out of the 10 nearest weather stations, the closest station with the least missing data is selected. In the next step, the app creates a digital mockup of the quilt. 

### Descriptive Statistics with pandas

Pandas is used to compute some descriptive statistics and I constructed a custom binning function based on the maximun and minimum temperatures from the dataset. Lambda functions are using to reformat the date and the binned maximum daily temperatures. (The binning function is based on the actual temperature values.) Then, the  dataframe is reshaped and missing data values are filled with a *null* value of 15. Each numerical value from 0 to 15 corresponds to a specific color. 

```python
 my_dates = noaa_data.DATE
    datetimes = my_dates.apply(
        lambda x: datetime.datetime.strptime(
            x,
            "%Y-%m-%d",
        )
    )
    months = datetimes.apply(lambda x: x.month)
    days = datetimes.apply(lambda x: x.day)
    noaa_data = noaa_data.assign(days=days)
    noaa_data = noaa_data.assign(months=months)
    my_levels = noaa_data.TMAX.apply(
        lambda x: grade_temp(
            noaa_data,
            int(x),
        )
    )
    noaa_data = noaa_data.assign(levels=my_levels)
    my_small = noaa_data[["months", "days", "levels"]]
    my_reshape = my_small.pivot(
        index="days",
        columns="months",
        values="levels",
    )
    my_reshape = my_reshape.fillna(15.0)
    return my_reshape
```
The next step is to tabulate the number of days in each bin and record the temperature ranges and insert this values into a data frame. 
This data frame is essentially a frequency distribution table. 
Dtreamlit.io displays the dataframe which is also recorded in the pdf pattern file. 
We use the pandas dataframe and the Pillow package create the digital  mock-up of the quilt. 
![jpg](/images/data_quilt_files/samplequilt.jpg)

### Actual fabric! 

The colors used in the mock up of the quilt are actual, commercially available fabric colors from Kona cotton, a commercially available fabric line. This brings my pdf pattern into line with industry standards for quilt patterns. Quilters can purchase the listed fabrics and create a quilt identical to the digital mockup. 

Here is the color range for your reference. 
![png](/images/data_quilt_files/ColorRange.PNG)

### Streamlit Dashboard

The frontend/dashboard is constructed using Streamlit.io. Streamlit displays each step of the process outlined above. The data can be inspected in the dashboard and the download button allows users to download a pdf with all the necessary information to create a physical version of the digital mockup. The app can be used to create multiple create mock ups based on year and zip code.  

## Conclusion 

This fun demonstration project has led to several other on-going projects that I hope to blog about soon. So if you're interested in displaying data in a community friendly way, keep following along!