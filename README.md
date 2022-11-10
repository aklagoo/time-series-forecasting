# Time Series Forecasting using Facebook Prophet

*Completed as a requirement for CSE 598: Modern Temporal Learning*

### Objective 

Use the data provided with this project. The data consists of 6 attributes measured daily over several years at several locations.

Forecast attribute x6 at location 6 for each of the next 20-time steps (days).

Use the methods described in the course. You may choose any method. You might start with a model which uses data from only the target attribute and target location first, and then expand your model with more attributes and more locations. Patterns for the target location might be similar to those at other locations, but maybe not all locations are similar.

## Data Description

The attributes x1 through x6 can be handled as continuous attributes, even though they are rounded to integers. Dates with data can vary slightly between locations (real data), but this is not expected to generate large impacts on models. These differences might be ignored to start, and enhanced later.