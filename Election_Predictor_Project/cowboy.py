import pandas as pd 
import numpy as np 
import os 
import csv 
from sqlalchemy import create_engine
import psycopg2

class wrangler():
    #This huge function cleans the csv's so they are ready for SQL upload
    def csv_cleaner(path,filename):
        """Description:
        All purpose csv cleaner. The primary function is to clean the columns so that they are SQL friendly. Removes any redundant
        or unnecessary columns that I don't want in the final csv, formats the data into the appropriate data type, and outputs the
        finished product to a new '_clean' csv.

        """
        def float_applier(x):
            try:
                if x.isnumeric() == True:
                    return float(x)
                else:
                    return x
            except AttributeError:
                return x

        temp_df = pd.read_csv(path,skiprows=9)
        #Drops columns with useless information
        temp_df = temp_df[[col for col in temp_df.columns if 'unnamed' not in col.lower()]].dropna()
        #Gets rid of all the special characters in the column names so they are SQL compatible
        temp_df.columns = [col.replace(' ','_').replace('-','_').replace('(','').replace(')','').replace('/','_').replace(',','').replace('+','_plus').replace('$','').replace('.','') for col in temp_df.columns]
        #Because the numerical columns are not obviously about Rent/Income etc. I will name them accordingly so they are easier to identify
        if filename.split('_')[1][:-4] in ['Rent','Income','Mortgage','FamilyIncome','DomesticWrk','HrsWrkd','EnglishProf']:
            temp_df.columns = [(filename.split('_')[1][:-4] + '_' + col) if col not in ['DivisionID','CED','CED_State','Census_year','Total'] else col for col in temp_df.columns]
        else:
            #This just makes sure a column starting with a number has a '_' so it's SQL compatible
            temp_df.columns = [('_' + col) if [c.isdigit() for c in col][0] else col for col in temp_df.columns]

        excl = ['DivisionID','CED','CED_State','Census_year','Total']

        #Simple renaming of columns
        temp_df.rename(columns={temp_df.columns[0]:'CED'},inplace=True)
        temp_df['CED'] = temp_df['CED'].apply(lambda x: x.replace('  (CED07)','').split(',')[0].lower())
        temp_df['Census_year'] = filename[:4]
        temp_df.reset_index(drop=True,inplace=True)

        for col in temp_df:
            temp_df[col] = temp_df[col].apply(float_applier)     

        temp_df['Sum_total'] = temp_df.loc[:,[col for col in temp_df.columns if col not in excl]].sum(axis=1)

        temp_df.drop(columns='Total',inplace=True)

        #Converts the column into a proportion, and drops the non-proportioned columns
        for col in temp_df:
            if temp_df[col].dtype == 'float64' or temp_df[col].dtype == 'int64' or np.issubdtype(temp_df[col].dtype, np.number) == True:
                if col != 'Census_year':
                    temp_df[col + '_pct'] = (temp_df[col]/temp_df['Sum_total'])*100
                    temp_df.drop(columns=col,inplace=True)

        temp_df.drop(columns='Sum_total_pct',inplace=True)
        temp_df.dropna(inplace=True)

        ref_table = pd.read_csv('./Database/AEC/All_reftable.csv')
        ref_table.rename(columns={'DivisionNm':'CED','StateAb':'CED_State'},inplace=True)
        ref_table.drop_duplicates(subset='CED',inplace=True)
        ref_table['CED'] = ref_table['CED'].apply(lambda x:x.lower())

        #Merging with the ref table, so each electorate has a unique DivisionID
        temp_df = ref_table[['DivisionID','CED','CED_State']].merge(temp_df,on='CED')
        #Naming and uploading to a new directory, ready to be put in my SQL table
        temp_string = './Database/ABS/clean_table_builder/' + filename[:-4] + '_clean' + '.csv'
        temp_df.to_csv(temp_string,index=False)

    #Function that uploads each csv based on category
    def sql_upload(name,directory):
        """Description:
        Uploads the csvs to the my SQL database. Creates a table based on the passed in csv columns and adds data to that table
        automatically. Sorts the tables by their category name, this is helped by my file naming convention.
        
        """
        #Sub function makes a sql query to create table based on the csv's column names.
        def table_maker(df,table_name):
            cols = df.columns
            data_types = df.dtypes
            #Automatically assigns data types based on the passed in df data type
            types_ = ['text' if type_ == 'object' else 'float' for type_ in data_types]
                
            #Starts the query string
            start_string = 'CREATE TABLE ' + table_name + ' ('
            empty_list = []
            #Adds in the name and type of the table
            for names,dtype in zip(cols,types_):
                empty_list.append(names)
                if len(empty_list) < len(cols):
                    col = names + ' ' + dtype + ','
                    start_string += col
                else:
                    col = names + ' ' + dtype
                    start_string += col
            
            #ends the string with the close bracket, which completes the sql query
            end_string = start_string + ')'
            return end_string

        #Creates a list of files under a category
        file_table = [file for file in os.listdir(directory) if name == file.split('_')[1]]
        df_t = pd.read_csv(directory + file_table[0])
        #Makes the sql query out of the first csv in the list of files under a specific category
        sql_string = table_maker(df_t,file_table[0].split('_')[1])
        conn_str = "host='localhost' \
        dbname='Election_DB' \
        user='postgres' \
        password='1234'"
        conn = psycopg2.connect(conn_str)
        cur = conn.cursor()
        #Creates the empty table with column names
        cur.execute(sql_string)
        #Populates the previously created empty table with respective data
        for tabs in file_table:
            with open(directory + tabs,'r') as f:
                next(f)
                cur.copy_from(f,file_table[0].split('_')[1],sep=',')
                conn.commit()
    
    def csv_interpolate_clean(path,filename):
        """Description:
        A modified version of the above csv cleaner function, similar methods, except not taking a proportion 
        so it can be interpolated later.
        
        """
        ref_table = pd.read_csv('./Database/AEC/All_reftable.csv')
        ref_table.rename(columns={'DivisionNm':'CED','StateAb':'CED_State'},inplace=True)
        ref_table.drop_duplicates(subset='CED',inplace=True)
        ref_table['CED'] = ref_table['CED'].apply(lambda x:x.lower())
        ref_table

        temp_df = pd.read_csv(path,skiprows=9)
        temp_df = temp_df[[col for col in temp_df.columns if 'unnamed' not in col.lower()]].dropna()
        temp_df.columns = [col.replace(' ','_').replace('-','_').replace('(','').replace(')','').replace('/','_').replace(',','').replace('+','_plus').replace('$','').replace('.','') for col in temp_df.columns]
        if filename.split('_')[1][:-4] in ['Rent','Income','Mortgage','FamilyIncome','DomesticWrk','HrsWrkd','EnglishProf']:
            temp_df.columns = [(filename.split('_')[1][:-4] + '_' + col) 
                            if col not in ['DivisionID','CED','CED_State','Census_year','Total'] else col for col in temp_df.columns]
        else:
            temp_df.columns = [('_' + col) if [c.isdigit() for c in col][0] else col for col in temp_df.columns]
            
        temp_df.rename(columns={temp_df.columns[0]:'CED'},inplace=True)
        temp_df['CED'] = temp_df['CED'].apply(lambda x: x.replace('  (CED07)','').split(',')[0].lower())
        temp_df.reset_index(drop=True,inplace=True)

        def float_applier(x):
            try:
                if x.isnumeric() == True:
                    return float(x)
                else:
                    return x
            except AttributeError:
                return x
        
        for col in temp_df.columns:
            temp_df[col] = temp_df[col].apply(float_applier)
        
        temp_df = ref_table[['DivisionID','CED','CED_State']].merge(temp_df,on='CED')
        temp_string = './Database/ABS/to_interpolate/' + filename[:-4] + '_interpolate' + '.csv'
        temp_df.to_csv(temp_string,index=False)

    #Imputer function that imputes data for the census data, corresponding to each election year
    def interpolater(low_file,up_file,directory,year):
        """Description:
            This interpolates data to align the census data with the election years so that they may be joined properly. It calculates
            a CAGR and applies it based on the difference between the specified interpolated year and the low bound. E.g. for the 2010
            election, it would calculate the 2006-2011 CAGR for all census stats, then apply that CAGR to the power of 4 from the 2006
            original census data.
        """
        #Defining a low/high (to impute between any two of the 3 census dates)
        low = pd.read_csv(directory + low_file)
        up = pd.read_csv(directory + up_file)
        
        #Merging the low and hi data so that the divisions line up for changing electorates or new divisions
        low_df = up[['DivisionID']].merge(low,on='DivisionID')
        up_df = low[['DivisionID']].merge(up,on='DivisionID')
        
        #Defining a cagr dataframe which will have the rates to apply
        cagr_df = low_df.copy(deep=True)
        #Defining a impute dataframe which the cagr dataframe's growth rates will be applied to
        impute_df = low_df.copy(deep=True)
        excl = ['DivisionID','CED','CED_State','Census_year','Total']
        #Calculating the CAGR based on the high/low dataframes
        for col in cagr_df.columns:
            #applying it to only the columns with numerical data
            if col not in excl:
                cagr_df[col] = ((up_df[col]/low_df[col])**(1/5) -1)
                cagr_df = cagr_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        #Applying the new growth figures after applying the cagr
        for col in impute_df.columns:
            if col not in excl:
                #Applies the CAGR to the lower file if the year is between the lower/upper bounds
                if float(year) > float(low_file.split('_')[0]) and float(year) < float(up_file.split('_')[0]):
                    impute_df[col] = impute_df[col]*(1 + cagr_df[col])**(abs(float(year) - float(low_file.split('_')[0])))

                #Applies the CAGR to the upper file if the year is greater than the upper bound (only for 2019 election)
                if float(year) > float(up_file.split('_')[0]):
                    impute_df[col] = up_df[col]*(1 + cagr_df[col])**(abs(float(year) - float(up_file.split('_')[0])))

                #Applies a negative CAGR to the lower files if the year is lower than the lower bound (only for 2004 election)
                if float(year) < float(low_file.split('_')[0]):
                    impute_df[col] = impute_df[col]*(1 - cagr_df[col])**(abs(float(year) - float(low_file.split('_')[0])))   

        #Makes a new column for the interpolated year based on the year argument that was passed in
        impute_df['Census_year'] = year

        for col in impute_df:
            #Only taking a proportion of the numerical data and ignoring any data with strings
            if impute_df[col].dtype == 'float64' or impute_df[col].dtype == 'int64' or impute_df[col][0].isnumeric()==True:
                #Ignoring census year and division id as they are numerical, but we don't want to take a proportion
                if col != 'Census_year' and col != 'DivisionID':
                    impute_df[col] = impute_df[col].apply(float)     

        #Makes a new total column based on the sum of all feature columns so that the proportion is balanced
        impute_df['Sum_total'] = impute_df.loc[:,[col for col in impute_df.columns if col not in excl]].sum(axis=1)

        #Droping the original total column
        impute_df.drop(columns='Total',inplace=True)

        #Calculating the proportion of the counts after imputing the growth from above
        for col in impute_df:
            #Only taking a proportion of the numerical data and ignoring any data with strings
            if impute_df[col].dtype == 'float64' or impute_df[col].dtype == 'int64' or impute_df[col][1].isnumeric()==True:
                #Ignoring census year and division id as they are numerical, but we don't want to take a proportion
                if col != 'Census_year' and col != 'DivisionID':
                    impute_df[col + '_pct'] = (impute_df[col]/impute_df['Sum_total'])*100
                    impute_df.drop(columns=col,inplace=True)

        #Dropping the sum total column as it is not needed anymore now that we have proportions
        impute_df.drop(columns='Sum_total_pct',inplace=True)
        #Dropping any residual nulls
        impute_df.dropna(inplace=True)

        #Naming the file and uploading it to the specified directory
        imp_string = './Database/ABS/interpolated/' + year + '_' + up_file.split('_')[1] + '_clean.csv'
        impute_df.to_csv(imp_string,index=False)
    
    def sql_upload_int(name,directory):
        """Description:
            A modified version of the original SQL uploader method. This will upload to existing SQL tables, so there's no need for the
            table maker function to be called again.
        """
        #Creates a list of files under a category
        file_table = [file for file in os.listdir(directory) if name == file.split('_')[1]]
        #Uploads the data to the respective table
        for tabs in file_table:
            with open(directory + tabs,'r') as f:
                next(f)
                conn_str = "host='localhost' \
                dbname='Election_DB' \
                user='postgres' \
                password='1234'"
                conn = psycopg2.connect(conn_str)
                cur = conn.cursor()
                cur.copy_from(f,file_table[0].split('_')[1],sep=',')
                conn.commit()   

    def sql_join(table,ttype):
        """Description:
            Similar to the table maker method, this builds a SQL query based on a list of table names, and joins all specified table
            names together, creating a joined data frame on the division id and the census/election year.
        """
        div = ' AND '.join([(table[0] + '.divisionid = ' + item + '.divisionid') for item in table[1:]])
        yr = ' AND '.join([(table[0] + '.' + ttype + '_year = ' + item + '.' + ttype + '_year') for item in table[1:]])
        sql_q = "SELECT * FROM " + ','.join(table) + " WHERE " + div + ' AND ' + yr
        return sql_q

    