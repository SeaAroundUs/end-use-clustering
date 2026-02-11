This readme file was generated on 19/08/2024 by Lauren A. Shea.
Last update: 05/03/2025


GENERAL INFORMATION

Title of Dataset: "Replication Data for: Spatial distribution of fishmeal and fish oil factories around the globe"

Lead/Corresponding Author Information
Name: Lauren A. Shea		
ORCID: 0000-0003-1201-0938
Institution: University of British Columbia
Address: Vancouver, BC, Canada

Contact Information
Name: Lauren A. Shea	
Current Institution: University of Victoria	
Current Address: Victoria, BC, Canada
Primary Email: laushea17@gmail.com

Supervisor/ Principal Investigator Information
Name: Rashid Sumaila
ORCID: 0000-0002-1851-1621
Institution: University of British Columbia
Address: Vancouver, BC, Canada
Email: r.sumaila@oceans.ubc.ca

Other Co-author Information:
Colette C. C. Wabnitz
Institution: University of British Columbia
Address: Vancouver, BC, Canada

William W. L. Cheung
Institution: University of British Columbia
Address: Vancouver, BC, Canada

Daniel Pauly
Institution: University of British Columbia
Address: Vancouver, BC, Canada

Date of data collection: February to April 2024 

Geographic location of data collection: Vancouver, Canada

Information about funding sources that supported the collection of the data: Packard Foundation Grant (2023-28472)


SHARING/ACCESS INFORMATION

Licenses/restrictions placed on the data: CC-BY (Attribution)

Links to publications that cite or use the data: Coming soon

Recommended citation for this dataset: Shea, L. A., Wabnitz, C. C. C., Cheung, W. W. L., Pauly, D., and Sumaila, U. R. 2024. "Replication Data for: Spatial distribution of fishmeal and fish oil factories around the globe". Borealis Data Repository. [Date accessed]. https://doi.org/10.5683/SP3/P0GLZF. 


DATA & FILE OVERVIEW

File List: 
FMFO_factories_raw_materials_v2.tab

METHODOLOGICAL INFORMATION

Description of methods used for collection/generation of data: 

We compiled the spatial database by first searching for FMFO production companies in each of the selected countries (n=63). Company names were found using national databases and membership associations (e.g., the European Fishmeal and Fish Oil Producers, IFFO, Marin Trust), peer-reviewed and grey literature, and online seafood business directories (e.g., www.trade-seafood.com). 

Our approach was informed by Clawson et al. (2022) and Oyinlola et al. (2018), two studies which investigated the spatial distribution of mariculture farms around the globe. The study covers all countries (n=63) that produced at least 500 mt of fish oil in 2022, according to data provided by IFFO: The Marine Ingredients Organization. These 63 countries produced 99.8% of global fish oil supply in 2022 (according to IFFO). 

We attempted to locate each company’s factories and verify their location using the satellite image layer in Google Maps. Verification involved zooming in to the presumed location and checking that there was a factory building there. Once verified, the coordinates of the plant(s) were recorded. In some cases, the factories could not be located due to the absence of a company’s website, lack of information on the company’s website, or factory addresses that could not be found on Google Maps. When a factory address was provided but could not be pinpointed, we used an open-source geocoding platform (https://geocode.xyz/) to geocode the address. 

We also searched company websites and MarinTrust factory certificates for information regarding raw materials and production capacity for individual factories. Production capacity data was sparse and excluded from dataset. Raw material data was included.


Methods for processing the data: This data was used to create spatial distribution maps in qGIS and visualizations in R Studio. 

Quality-assurance procedures performed on the data: Coordinate data was categorized into one of four data types to describe the level of detail and certainty in each set of coordinates. See descriptions for data types below in "Data-Specific Information".

People involved with sample collection, processing, analysis and/or submission: 
Lauren A. Shea
U. Rashid Sumaila
Colette C. C. Wabnitz
Daniel Pauly
William W. L. Cheung


DATA-SPECIFIC INFORMATION FOR: FMFO_factories_raw_materials_v2.tab

Number of variables: 14

Number of cases/rows: 1189

Variable List: 

factory_id - A unique numeric identifier for each unique factory (n=506).
country - Name of country where factory is located.
lat - Latitude for factory location (WGS84 coordinate system).
long - Longitude for factory location (WGS84 coordinate system).
company - Name of the company that owns the factory.
data_type - Data type refers to the quality of each piece of information, from A to D. See descriptions below:
A - Site location verified using the Google Maps satellite image layer. Factory address published on company website, provided by third-party source (e.g. Marin Trust certificate), or an active 'Business' pin on Google Maps.
B - Site address published on company website or by a third-party source but not verified using satellite image layer. Address was geocoded using the open-source Geocode.xyz platform and the geocode confidence score was greater than 0.5.
C - The city or province in which the site is located is known and the coordinates for the city/province are used OR the geocode confidence score for the address is 0.5 or below (typically depicting that the address could only be matched to the city/province level.
D - Only the country in which the factory is located is known.

material_type - Describes the type of raw material that the factory is using. This could be whole (i.e., whole fish), fishery by-products (i.e., fish trimmings or waste from capture fisheries), aquaculture by-products (i.e., fish trimmings or waste from fish farms), or misc by-products (these by-products were not specified as to whether they originated from fisheries or aquaculture). 
material_type_simple - Related to the previous variable (material_type) but all by-products are classified as the same.
comm_name - Common name that the fish species is referred to as by the factory/company's website.
sci_name - If not provided by factory/company, the common name was matched to its scientific name on fishbase.org.
func_group - Functional group for each fish species according to seaaroundus.org.
source - Organization name and/or link to article where information for each factory was found. 
website - FMFO company website.
date - Month/Year of data collection.



Missing data codes: NA
