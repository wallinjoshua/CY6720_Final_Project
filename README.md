# Predicting U.S. High School Student Graduation Outcomes Using Two Longitudinal Studies

## Project Abstract
High Schools in the United States, as well as state and federal education agencies, have a vested interest in ensuring that students graduate on time. Often, individual schools and districts have limited resources to support students that are at risk of failing, and look to identify these students early-on, with the hope of remediating any educational issues before they lead to a negative outcome. In this project, we focus on a public dataset tracking high school students from 2002 to 2004. We extract features from these datasets and build a model to predict whether or not a student in 10th grade will later graduate ontime (i.e., at or before Spring 2004). We apply various techniques to simplify this model and improve its accuracy in the presence of imbalanced classes. Finally, we attempt to apply the model to data from a similar longitudinal study beginning in 2009.

## Repository File Structure
| Folder Name | Description |
|-------------|-------------|
| Column Files | The set of files that can be used with extract_data.py to select a set of relevant features from a csv data file |
| Data Extraction & Processing | Scripts to extract data from large data files and a preliminary attempt to capture how HSLS:2009 data features may be converted to commensurate ELS:2002 features |
| Datasets | Copies of **some** of the datasets used for this analysis, along with extracted data relevant to particular modeling activities. All full datasets may be found at the websites listed below. |
| Figures | Graphs generated using pyplot during the course of analysis |
| Final Report | Tex files that may be compiled to access the final report document (also submitted via GradeScope) |
| Model Code | The code for various modeling attempts and steps, from the initial tests (using all 4 possible algorithms), to final tests that applied undersampling and minimized the number of features to the AdaBoost model |
| Saved Models | Load-able dumps of various models, that may be reloaded to run tests. Model names are self-explanatory |
| Saved Output | Stored output that was used for generating graphs and as a record of the results recorded in the final report |

Link to 2002 Longitudinal Study: [ELS:2002](https://nces.ed.gov/surveys/els2002/)  
Link to 2009 Longitudinal Study: [HSLS:2009](https://nces.ed.gov/surveys/hsls09/)  