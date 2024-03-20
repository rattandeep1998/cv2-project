import pandas as pd
import deplot_metrics as deplot_metrics
import json

# ChartQA
# targets = ["Characteristic,Inhabitants in millions\n2025*,47.28\n2024*,47.11\n2023*,46.93\n2022*,46.74"]
# predictions = ["Characteristic | Inhabitants in millions & 2025* | 47.28 & 2024* | 47.11 & 2023* | 46.93 & 2022* | 46.74 & 2021* | 46.54 & 2020 | 46.49 & 2019 | 46.49 & 2018 | 46.45 & 2017 | 46.41 & 2016 | 46.4 & 2015 | 46.41 & 2014 | 46.46 & 2013 | 46.59 & 2012 | 46.77 & 2011 | 46.74 & 2010 | 46.56"]

targets = ["""title | my table
year | argentina | brazil
1999 | 200 | 158
"""]

predictions = ["""title | my table
year | argentina | brazil
1999 | 202 | 0
"""]

predictions1 = ["""title | my table
argentina | brazil | time
200 | 158 | 1999
"""]

# Vistext
# targets = ["Digital share of overall music sales in selected countries from 2004 to 2014 <s> Year Germany Dec 31, 2003 0.01 Dec 31, 2004 0.027 Dec 31, 2005 0.046 Dec 31, 2006 0.058 Dec 31, 2007 0.08 Dec 31, 2008 0.101 Dec 31, 2009 0.126 Dec 31, 2010 0.153 Dec 31, 2011 0.191 Dec 31, 2012 0.206 Dec 31, 2013 0.22"]
# predictions = ["Characteristic | [unnamed data series #0] | [unnamed data series #1] & 2004.0 | 0.01 | 0.01 & 2006.0 | 0.05 | 0.05 & 2008.0 | 0.08 | 0.08 & 2010.0 | 0.13 | 0.13 & 2012.0 | 0.19 | 0.19"]

print(deplot_metrics.table_datapoints_precision_recall(targets, predictions1))
print(deplot_metrics.table_number_accuracy(targets, predictions1))

# Predictions1 is more close to Target -- But Precision and Recall score with Predictions1 is less than that with Predictions