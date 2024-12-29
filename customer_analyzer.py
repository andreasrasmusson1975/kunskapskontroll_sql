"""
module customer_analyser.py

This module provides functionality for segmenting the existing customers based on available
demographic data and identify segments that are profitable even at a 40% discount rate.

Author: Andreas Rasmusson
Date: December 13, 2024

Usage:
    from customer_analyzer import CustomerAnalyzer
    ca=CustomerAnalyzer(f"mssql+pyodbc://localhost/AdventureWorks2022?driver=ODBC+Driver+18+for+SQL+Server&trusted_connection=yes&TrustServerCertificate=yes")
    ca.plot_top_five_segments()
    ca.plot_top_five_products()
Classes:

    CustomerAnalyzer - Provides all the functionality - connection to database, needed queries and plotting functions

"""

# Perform necessary imports
from sqlalchemy import create_engine,text
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from tqdm.notebook import tqdm

# Set seaborn theme and context
sns.set_theme(style="darkgrid")
sns.set_context(context="notebook")

# A CTE query for creating  the following binary demographic variables for each customer:
#
#   1. AgeLower         1 if the customer is below 50 years old and 0 otherwise
#   2. Married          1 if the customer is married and 0 otherwise
#   3. EducationLower   1 if the education level of the customer is
#                         "Partial High school" or "High school" and 0 otherwise
#   4. IncomeLower      1 if the income level of the customer is "0-25000" or
#                         "25001 - 50000" and 0 otherwise
#   5. GenderFemale     1 if the customer is female and 0 otherwise
#   6. HomeOwner        1 if the customer is a homeowner and 0 otherwise
#   7. CarOwner         1 if the customer owns a car and 0 otherwise
#   8, ChildrenAtHome   1 if the customer has children at home and 0 otherwise
#
# For this query, we use the Sales.Customer and Person.Person tables. In Person.Person, there is
# a Demographics xml column that we can extract the demographic data from.
sql_demographics = """
WITH XMLNAMESPACES ('http://schemas.microsoft.com/sqlserver/2004/07/adventure-works/IndividualSurvey' AS ns), -- Alias for the Url
/*
    Step 1: Extract demographic data
*/
CustomerDemographics AS(
SELECT
	 T1.CustomerID
	,T2.BusinessEntityID
	,T2.Demographics.value('(/ns:IndividualSurvey/ns:BirthDate)[1]', 'DATE') AS BirthDate -- Such is the syntax for extracting values
	,T2.Demographics.value('(/ns:IndividualSurvey/ns:MaritalStatus)[1]', 'NVARCHAR') AS MaritalStatus
	,T2.Demographics.value('(/ns:IndividualSurvey/ns:YearlyIncome)[1]', 'NVARCHAR(MAX)') AS YearlyIncome
	,T2.Demographics.value('(/ns:IndividualSurvey/ns:Gender)[1]', 'NVARCHAR') AS Gender
	,T2.Demographics.value('(/ns:IndividualSurvey/ns:TotalChildren)[1]', 'INTEGER') AS TotalChildren
	,T2.Demographics.value('(/ns:IndividualSurvey/ns:NumberChildrenAtHome)[1]', 'INTEGER') AS NumberChildrenAtHome
	,T2.Demographics.value('(/ns:IndividualSurvey/ns:Education)[1]', 'NVARCHAR(MAX)') AS Education
	,T2.Demographics.value('(/ns:IndividualSurvey/ns:Occupation)[1]', 'NVARCHAR(MAX)') AS Occupation
	,T2.Demographics.value('(/ns:IndividualSurvey/ns:HomeOwnerFlag)[1]', 'INTEGER') AS HomeOwnerFlag
	,T2.Demographics.value('(/ns:IndividualSurvey/ns:NumberCarsOwned)[1]', 'INTEGER') AS NumberCarsOwned
	,T2.Demographics
FROM
    Sales.Customer T1
INNER JOIN
    Person.Person T2
ON
    T1.PersonID = T2.BusinessEntityID
WHERE
    T2.PersonType = 'IN' -- We only want private individuals
AND
    TerritoryID <= 5 -- We only want US customers
)
/*
    Last step: Convert the demographic data to binary variables.
*/
SELECT
	 CustomerID
	,BusinessEntityID
	,CASE WHEN DATEDIFF(YEAR, BirthDate, GETDATE()) <= 50 THEN '1' ELSE '0' END AgeLower
	,CASE WHEN MaritalStatus = 'M' THEN '1' ELSE '0' END Married
    ,CASE WHEN Education IN ('Partial High School','High School') THEN '1' ELSE '0' END EducationLower
    ,CASE WHEN YearlyIncome IN ('0-25000','25001-50000') THEN '1' ELSE '0' END IncomeLower
    ,CASE WHEN Gender = 'F' THEN '1' ELSE '0' END GenderFemale
	,CAST(HomeOwnerFlag AS NVARCHAR) HomeOwner
	,CASE WHEN NumberCarsOwned > 0 THEN '1' ELSE '0' END CarOwner
	,CASE WHEN NumberChildrenAtHome > 0 THEN '1' ELSE '0' END ChildrenAtHome
FROM
	CustomerDemographics
;
"""

# A query where we simply collect all the orders from the
# Sales.SalesOrderHeader table.
sql_orders ="""
SELECT
     CustomerID
	,SalesOrderID
	,OrderDate
FROM	
    Sales.SalesOrderHeader T2
;
"""

# A query for fetching the reason of sale for each order.
# Here we use the Sales.SalesOrderHeader and 
# Sales.SalesOrderHeaderSalesReason tables. We also create
# a binary variable SalesReasonIDIsAppropriate which is
# 1 if at least one of the reasons for the order  was "Price"
# and/or "Promotion". 
sql_sales_reason = """
SELECT
	  T1.SalesOrderID
      /*
            There can be several sales reasons for one order. Here, we are only
            interested in those orders for which at least one of the  reasons for 
            the sale was "Price" and/or "Promotion". This is why we take the max here.    
      */
	 ,MAX(CASE WHEN ISNULL(T2.SalesReasonID,-1) = 2 OR ISNULL(T2.SalesReasonID,-1) = 1 THEN 1 ELSE 0 END) SalesReasonIDIsAppropriate
FROM
	Sales.SalesOrderHeader T1
LEFT JOIN
	Sales.SalesOrderHeaderSalesReason T2
ON
	T1.SalesOrderID = T2.SalesOrderID
GROUP BY
	T1.SalesOrderID
;
"""

# In this query we simple fetch content from
# the Sales.SalesOrderDetail table.
sql_order_details = """
SELECT
     SalesOrderID
	,SalesOrderDetailID
	,OrderQty
	,ProductID
	,UnitPrice
	,UnitPriceDiscount
	,LineTotal
FROM
	Sales.SalesOrderDetail
;
"""

# Query for fetching content from the Production.Product and
# Production.ProductSubCategory tables.
sql_products = """
SELECT
     T1.ProductID
	,T1.StandardCost
    ,T1.Name ProductName
    ,T3.Name ProductCategoryName
    ,T2.Name ProductSubCategoryName
FROM
	Production.Product T1
LEFT JOIN
    Production.ProductSubCategory T2
ON
    T1.ProductSubcategoryID = T2.ProductSubcategoryID
LEFT JOIN
    Production.ProductCategory T3
ON
    T2.ProductCategoryID = T3.ProductCategoryID
;
"""

# A no content Exception class. We just need to
# define the name of the exception (MergeException)
# this is part of the validation process and we 
# will throw this exception if certain validations
# do not pass.
class MergeException(Exception):
    pass


class CustomerAnalyzer :
    """
    A class for segmenting the existing customers based on available
    demographic data and identify segments that are profitable even at 
    a 40% discount rate.

    Attributes:
        demogr_df:              pandas.core.frame.DataFrame
        orders_df:              pandas.core.frame.DataFrame
        sreason_df:             pandas.core.frame.DataFrame
        orddet_df:              pandas.core.frame.DataFrame
        prod_df:                pandas.core.frame.DataFrame
        demord_df:              pandas.core.frame.DataFrame
        demordsreason_df:       pandas.core.frame.DataFrame
        demordsreasonorddet_df: pandas.core.frame.DataFrame
        detailed_data_df:       pandas.core.frame.DataFrame
        data_df:                pandas.core.frame.DataFrame
        aggregated_data_df:     pandas.core.frame.DataFrame

    """
    def __init__(self,connection_string: str):
        self.engine = create_engine(connection_string)
        self.__get_dfs__()

    # Create all the dataframes needed for the analysis
    def __get_dfs__(self):
        with self.engine.connect() as conn:
            # fetc needed data from the AdventureWorks database
            self.demogr_df = pd.read_sql(sql_demographics,conn).reset_index(drop=True)
            self.orders_df = pd.read_sql(sql_orders,conn).reset_index(drop=True)
            self.sreason_df = pd.read_sql(sql_sales_reason,conn).reset_index(drop=True)
            self.orddet_df = pd.read_sql(sql_order_details,conn).reset_index(drop=True)
            self.prod_df = pd.read_sql(sql_products,conn).reset_index(drop=True)
            # merge dataframes in a controlled way so that we can perform validation
            # as we go.
            self.__demogr_merge_orders__()
            self.__demord_merge_sreason__()
            self.__demordsreason_merge_orddet__()
            self.__create_data_df_and_detailed_data_df__()
            self.__create_aggregated_data_df__()
    
    # Merge the demogr_df and orders_df dataframes 
    # to form the new dataframe demord_df and perform 
    # validation
    def __demogr_merge_orders__(self):
        self.demord_df = self.demogr_df.merge(
            self.orders_df,
            how='inner',
            on='CustomerID'
        )
        # Perform validation
        if self.demord_df['CustomerID'].unique().shape[0] != self.demogr_df.shape[0]:
            raise MergeException("Number of customer ids changed in the merge between demogr_df and orders_df. This shouldn't happen.")
        
    # Merge the demord_df and sreason_df to form the
    # new dataframe demordsreason_df and perform 
    # validation
    def __demord_merge_sreason__(self):
        self.demordsreason_df=self.demord_df.merge(
            self.sreason_df,
            how='inner',
            on='SalesOrderID'
        )
        # Perform validation
        if self.demordsreason_df['CustomerID'].unique().shape[0] != self.demord_df['CustomerID'].unique().shape[0]:
            raise MergeException(
                "Number of customer ids changed in the merge between demord_df and sreason_df. This shouldn't happen"
            )
        if self.demordsreason_df['SalesOrderID'].unique().shape[0] != self.demord_df['SalesOrderID'].unique().shape[0]:
            raise MergeException(
                "Number of order ids changed in the merge between demord_df and sreason_df. This shouldn't happen"
            )
   
    # Merge the demordsreason_df and orddet_df to form
    # the new dataframe demordsreasonorddet_df and perform
    # validation 
    def __demordsreason_merge_orddet__(self):
        self.demordsreasonorddet_df = self.demordsreason_df.merge(
            self.orddet_df,
            how='inner',
            on='SalesOrderID'
        )
        # Perform validation
        if self.demordsreasonorddet_df['CustomerID'].unique().shape[0] != self.demordsreason_df['CustomerID'].unique().shape[0]:
            raise MergeException(
                "Number of Customer ids changed in the merge between demordsreason_df and orddet_df. This shouldn't happen"
            )
        if self.demordsreasonorddet_df['SalesOrderID'].unique().shape[0] != self.demordsreason_df['SalesOrderID'].unique().shape[0]:
            raise MergeException(
                "Number of order ids changed in the merge between demordsreason_df and orddet_df. This shouldn't happen"
            )
        
    # Merge the demordsreasonorddet_df and prod_df dataframe
    # to form the new dataframe data_df ad perform validation.
    # Then add calculated columns Profit and AdjustedProfit.
    # Then compress the binary demographics variables to a 
    # single Label column. Finally, create the dataframe
    # detailed_data_df and clean up the data_df dataframe
    # to only hold necessary columns
    def __create_data_df_and_detailed_data_df__(self):
        self.data_df = self.demordsreasonorddet_df.merge(
            self.prod_df,
            how='inner',
            on='ProductID'
        )

        # Perform validation
        if self.data_df['CustomerID'].unique().shape[0] != self.demordsreasonorddet_df['CustomerID'].unique().shape[0]:
            raise MergeException(
                "Number of Customer ids changed in the merge between demordsreasonorddet_df and prod_df. This shouldn't happen"
            )
        if self.data_df['SalesOrderID'].unique().shape[0] != self.demordsreasonorddet_df['SalesOrderID'].unique().shape[0]:
            raise MergeException(
                "Number of ids changed in the merge between demordsreasonorddet_df and prod_df. This shouldn't happen")
        
        # Add calculated columns
        self.data_df['Profit']=self.data_df['LineTotal']-self.data_df['StandardCost']
        self.data_df['AdjustedProfit']=0.6*self.data_df['LineTotal']-self.data_df['StandardCost']
        
        # compress the binary demographics variables to a 
        # single Label column
        labels = [
            'Married',
            'EducationLower',
            'IncomeLower',
            'GenderFemale',
            'HomeOwner',
            'CarOwner',
            'ChildrenAtHome'
        ]
        label_series=self.data_df['AgeLower'].copy()
        for label in labels:
            label_series+=self.data_df[label]
        self.data_df['Label']=label_series
        
        # Create the detailed_data_df dataframe
        self.detailed_data_df=self.data_df.copy()
        self.detailed_data_df.drop(columns=[
            'AgeLower',
            'Married',
            'EducationLower',
            'IncomeLower',
            'GenderFemale',
            'HomeOwner',
            'CarOwner',
            'ChildrenAtHome'
        ], inplace=True)
        self.detailed_data_df.reset_index(inplace=True,drop=True)

        # Clean up the data_df dataframe
        self.data_df=self.detailed_data_df.loc[
            :,
            [
                'Label',
                'CustomerID',
                'SalesOrderID',
                'SalesOrderDetailID',
                'ProductID',
                'SalesReasonIDIsAppropriate',
                'AdjustedProfit'
            ]
        ].sort_values(by='Label')
        self.data_df.reset_index(inplace=True,drop=True)
    
    # Create the aggregated_data_df dataframe by grouping
    # the data_df dataframe on distinct labels (customer
    # segments) and then calculate the following quantities
    # within each segment:
    #
    # 1. Number of distinct customers
    # 2. Number of distinct orders
    # 3. Numer of products ordered
    # 4. Number of distinct products
    # 5. The average of the SalesReasonIDIsAppropriate column
    # 6. The average of the AdjustedProfit column
    # 
    # Then filter the aggregated dataframe so that we only have
    # large enough segments to ensure CLT applicability. Then
    # add confidence interval lower bounds for the point 
    # estimates SegmentAverageSalesReasonIDIsAppropriate and
    # SegmentAverageAdjustedProfit. Then filter the dataframe
    # again so that we only have segments with a high lower  
    # bound for the SegmentAverageSalesReasonIDIsAppropriate
    # point estimate and positive lower bound for the 
    # AverageAdjustedProfit estimate. Finally, perform sorting 
    # and index reset.
    def __create_aggregated_data_df__(self):
        # Aggregate on the order level
        df=self.data_df.copy().groupby(
            by=['Label','CustomerID','SalesOrderID'],                
            as_index=False
        ).agg(
            NumOrderDetail = ('SalesOrderDetailID','count'),
            AvgSalesReasonIDIsAppropriate = ('SalesReasonIDIsAppropriate','mean'),
            AvgAdjustedProfit = ('AdjustedProfit','mean')
        )
        # Aggregate on the customer level
        df = df.groupby(
            by=['Label','CustomerID'],
            as_index=False
        ).agg(
            NumOrderDetail = ('NumOrderDetail','sum'),
            NumDistinctOrders = ('SalesOrderID',lambda x:len(set(x))),
            NumDistinctCustomers = ('CustomerID',lambda x:len(set(x))),
            AvgAvgSalesReasonIDIsAppropriate = ('AvgSalesReasonIDIsAppropriate','mean'),
            AvgAvgAdjustedProfit = ('AvgAdjustedProfit','mean')
        )
        # Aggregate on the label level
        df2 = df.groupby(
            by='Label',
            as_index=False
        ).agg(
            SegmentNumOrderDetail = ('NumOrderDetail','sum'),
            SegmentNumDistinctOrders = ('NumDistinctOrders','sum'),
            SegmentNumDistinctCustomers = ('NumDistinctCustomers','sum'),
            SegmentAvgAvgAvgSalesReasonIDIAppropriate = ('AvgAvgSalesReasonIDIsAppropriate','mean'),
            SegmentAvgAvgAvgAdjustedProfit = ('AvgAvgAdjustedProfit','mean')
        )

        # Add confidence interval lower bounds
        df2 = self.__set_conservative_lower_bounds__(
            df,
            df2,
            'AvgAvgSalesReasonIDIsAppropriate',
            'Label'
        )
        df2 = self.__set_conservative_lower_bounds__(
            df,
            df2,
            'AvgAvgAdjustedProfit',
            'Label'
        )

        # Filter
        df2 = df2[df2['SegmentNumDistinctCustomers'] >= 50]
        df2=df2[df2['AvgAvgAvgSalesReasonIDIsAppropriateConservativeCILowerBound'] >= 0.7]

        # Sort and reset index
        df2.sort_values(
            by='AvgAvgAvgAdjustedProfitConservativeCILowerBound',
            ascending=False,
            inplace=True
        )
        df2.reset_index(inplace=True,drop=True)
        
        self.aggregated_data_df_customer_level = df
        self.aggregated_data_df = df2

    def __set_conservative_lower_bounds__(
        self,
        base_df: pd.core.frame.DataFrame,
        agg_df: pd.core.frame.DataFrame,
        variable: str,
        label: str
    ) -> pd.core.frame.DataFrame:
        agg_df = self.__set_ci_lower_bounds__(
            base_df,
            agg_df,
            variable,
            label
        )
        agg_df = self.__set_bootstrapped_lower_bounds__(
            base_df,
            agg_df,
            variable,
            label
        )
        agg_df['Avg'+variable+'ConservativeCILowerBound'] = np.minimum(agg_df['Avg'+variable+'CILowerBound'],agg_df['Avg'+variable+'BootstrappedCILowerBound'])
        agg_df.drop(['Avg'+variable+'CILowerBound','Avg'+variable+'BootstrappedCILowerBound'],axis=1,inplace=True)
        return agg_df

    # Helper method that, given a base dataframe with a numerical
    # column, an aggregated dataframe for that numerical column
    # and a label column, calculates confidence interval lower
    # bounds for each distinct label in the label column  
    def __set_ci_lower_bounds__(
            self,
            base_df: pd.core.frame.DataFrame,
            agg_df: pd.core.frame.DataFrame,
            variable: str,
            label: str
    ) -> pd.core.frame.DataFrame:
        
        # Create an empty dataframe with two columns:
        #
        # 1. The label column
        # 2. The condfidence interval lower bound column  
        ci_df=pd.DataFrame(columns=[label,'Avg'+variable+'CILowerBound'])
        
        # Iterate over the rows of the aggregated dataframe and fill
        # the newly created dataframe with confidence interval lower
        # bounds.
        for idx,row in agg_df.iterrows():
            observations = base_df[base_df[label]==row[label]][variable].copy()
            average = observations.mean()
            standard_error = np.std(observations,ddof=1)/np.sqrt(len(observations))
            l=average - 1.645 * standard_error
            ci_df.loc[idx] = [row[label],l]
        # Merge the aggregated dataframe with the newly
        # created dataframe
        agg_df=agg_df.merge(ci_df,how='inner',on=label)
        return agg_df

    def __set_bootstrapped_lower_bounds__(
            self,
            base_df: pd.core.frame.DataFrame,
            agg_df: pd.core.frame.DataFrame,
            variable: str,
            label: str,
            n_iter: int = 10000
        ):

        # Create an empty dataframe with two columns:
        #
        # 1. The label column
        # 2. The condfidence interval lower bound column  
        ci_df=pd.DataFrame(columns=[label,'Avg'+variable+'BootstrappedCILowerBound'])

        # Iterate over the rows of the aggregated dataframe and fill
        # the newly created dataframe with confidence interval lower
        # bounds.
        lst = [(idx,row) for idx,row in agg_df.iterrows()]
        for idx,row in tqdm(lst):
            observations = base_df[base_df[label]==row[label]][variable].copy()
            avgs = [observations.sample(n=len(observations),replace=True).mean() for i in range(n_iter)]
            l = np.percentile(avgs,5)
            ci_df.loc[idx] = [row[label],l]
        
        # Merge the aggregated dataframe with the newly
        # created dataframe
        agg_df=agg_df.merge(ci_df,how='inner',on=label)
        return agg_df
    
    def get_top_products(self,label: str) -> pd.core.frame.DataFrame:
        """
        A method that creates and returns a dataframe with sales volume and
        profit data for a given market segment. Only products with a sales
        volume larger than 50 is included.

        Arguments:
            label: str (the market segment label)
        Returns:
            df: pd.core.frame.DataFrame
        """
        
        # Create the df with the desired quantities
        df=self.detailed_data_df[self.detailed_data_df['Label'] == label].groupby(
            by=['CustomerID','ProductSubCategoryName'],
            as_index=False
        ).agg(
            NumSold = ('SalesOrderDetailID','count'),
            NumDistinctCustomers = ('CustomerID',lambda x:len(set(x))),
            AvgAdjProfit=('AdjustedProfit','mean')
        )
        
        df2 = df.groupby(
            by='ProductSubCategoryName',
            as_index=False
        ).agg(
            NumSold = ('NumSold','sum'),
            NumDistinctCustomers = ('NumDistinctCustomers','sum'),
            AvgAvgAdjProfit = ('AvgAdjProfit','mean')
        )
        
        # Calculate confidence interval lower bounds for
        # each product sub category
        df2=self.__set_conservative_lower_bounds__(df,df2,'AvgAdjProfit','ProductSubCategoryName')
        # Filter the dataframe on the number of customers
        df2=df2[df2['NumDistinctCustomers'] >= 50]

        # Clean up
        df2.sort_values(by='NumSold',ascending=False,inplace=True)
        df2.reset_index(inplace=True,drop=True)
        
        return df2
    
    def plot_top_five_segments(self):
        """
        A metod for plotting the top 5 customer segments
        in terms of confidence interval lower bounds for
        the expected profit per sold product.
        """

        # Create a dataframe consisting of the data
        # that is to be plotted
        df = self.aggregated_data_df.sort_values(
            by='AvgAvgAvgAdjustedProfitConservativeCILowerBound',
            ascending=False
        ).iloc[:5,:]
        
        # Create a bar chart
        fig,ax=plt.subplots(figsize=(10,5))
        sns.barplot(
            data=df, 
            x='AvgAvgAvgAdjustedProfitConservativeCILowerBound', 
            y=['1','2','3','4','5'],
            palette = sns.light_palette("navy", reverse=False, as_cmap=False,n_colors=5),
            edgecolor=".2",
            hue='AvgAvgAvgAdjustedProfitConservativeCILowerBound',
            legend=False
        )
        # Set axis labels and title
        ax.set_xlabel('Profitability Score Lower Bound')
        ax.set_ylabel('Segment')
        ax.set_title('TOP 5 SEGMENTS AND THEIR PROFITABILITY SCORE LOWER BOUNDS')
        
        # Create a legend containging descriptions of the segment labels in the chart
        legend_entries = ''.join([f'{idx+1}: {self.label_to_description(label)}\n' for idx,label in df['Label'].items()])
        plt.annotate(
            legend_entries,
            xy=(0.0, -0.3),
            xycoords='axes fraction',
            ha='left',
            va='center',
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", edgecolor="black", facecolor="white")
        )
        
        # Perform a bit of trickery to make things look well placed
        ax.set_position([0.2, 0.2, 0.7, 0.6])
        plt.tight_layout()
        
        # Show the image
        plt.show()

    def plot_top_five_products(self,label: str):
        """
        A method for plotting, for a specific customer segment, the top selling products
        sub categories within that segments in terms of the number of sold products. 
        Two bar charts are produced: One with the categories and their respective confidence 
        interval lower bounds for the expected profit per sold product, and one with the
        categories and their respective total volume of sales.

        Arguments:
            label: str (the label identifying the segment)
        """
        
        # Create the dataframe containing the data that is to
        # be plotted
        df=self.get_top_products(label)
        
        #Create the plots
        fig,axes=plt.subplots(1,2,figsize=(15,5))
        sns.barplot(
            data=df.sort_values(by='AvgAvgAdjProfitConservativeCILowerBound',ascending=False).iloc[:5,:],
            x='AvgAvgAdjProfitConservativeCILowerBound',
            y='ProductSubCategoryName',
            palette = sns.light_palette("navy", reverse=False, as_cmap=False,n_colors=5),
            edgecolor=".2",
            hue='AvgAvgAdjProfitConservativeCILowerBound',
            legend=False,
            ax=axes[0]
        )
        sns.barplot(
            data=df.sort_values(by='AvgAvgAdjProfitConservativeCILowerBound',ascending=False).iloc[:5,:],
            x='NumSold',
            y='ProductSubCategoryName',
            palette = sns.light_palette("navy", reverse=False, as_cmap=False,n_colors=5),
            edgecolor=".2",
            hue='AvgAvgAdjProfitConservativeCILowerBound',
            legend=False,
            ax=axes[1]
        )
        # Set axis labels and titles
        axes[0].set_xlabel('Profitability Score Lower Bound')
        axes[0].set_ylabel('Product')
        axes[0].set_title('TOP 5 SELLERS AND PROFITABILITY SCORE LOWER BOUNDS (TARGETED SEGMENT)')
        axes[1].set_xlabel('Number of items sold')
        axes[1].set_ylabel('Product')
        axes[1].set_title('TOP 5 SELLERS AND THEIR TOTAL VOLUME OF SALES (TARGETED SEGMENT)')
        
        # Perform a bit of trickery to make things look well placed
        plt.tight_layout()

        # Show the image
        plt.show()
    
    def label_to_description(self,label: str) -> str:
        """
        A method for getting the description for a
        label (market segment label)

        Arguments:
            label: str
        Returns:
            description: str
        """
        description = ''
        if label[0] == '1':
            description+='Age 50 or below'
        else:
            description+='Above age 50'
        if label[1] == '1':
            description+=', married'
        else:
            description+=', unmarried'
        if label[2] == '1':
            description+=', low education'
        else:
            description+=', high education'
        if label[3] == '1':
            description+=', low income'
        else:
            description+=', high income'
        if label[4] == '1':
            description+=' women who '
        else:
            description+=' men who '
        if label[5] == '1':
            description+='own a home and '
        else:
            description+='do not own a home and '
        if label[6] == '1':
            description+='own a car, and who '
        else:
            description+='do not own a car, and who '
        if label[7] == '1':
            description+='have children at home.'
        else:
            description+='do not have children at home.'
        return description

    
    


   


    

    