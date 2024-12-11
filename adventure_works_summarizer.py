from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")  
SERVER = 'localhost'
DATABASE = 'AdventureWorks2022'
SQL_CONNECTION_STRING = f"mssql+pyodbc://localhost/AdventureWorks2022?driver=ODBC+Driver+18+for+SQL+Server&trusted_connection=yes&TrustServerCertificate=yes"
engine = create_engine(SQL_CONNECTION_STRING)

class AdventureWorksSummarizer:
    def __init__(self):    
        self.__SERVER__ = 'localhost'
        self.__DATABASE__ = 'AdventureWorks2022'
        self.__SQL_CONNECTION_STRING__ = f"mssql+pyodbc://{self.__SERVER__}/{self.__DATABASE__}?driver=ODBC+Driver+18+for+SQL+Server&trusted_connection=yes&TrustServerCertificate=yes"
        self.engine = create_engine(self.__SQL_CONNECTION_STRING__)
        self.schema_tables_views_df, self.view_counts_df, self.table_counts_df = self.__get_schema_table_view_dfs__()
    
    def __get_schema_table_view_dfs__(self):
        with self.engine.connect() as conn:
            sql="""
            SELECT 
                T2.NAME SCHEMA_NAME
                ,T1.NAME TABLE_NAME
                ,TRIM(T1.TYPE) AS OBJECT_TYPE
                ,SUM(T3.ROW_COUNT) AS ROW_COUNT
            FROM 
                SYS.OBJECTS T1
            JOIN 
                SYS.SCHEMAS T2 
            ON 
                T1.SCHEMA_ID = T2.SCHEMA_ID
            LEFT JOIN 
                SYS.DM_DB_PARTITION_STATS T3 
            ON 
                T1.OBJECT_ID = T3.OBJECT_ID
            WHERE 
                T1.TYPE IN ('U', 'V')
            AND 
                (T3.INDEX_ID IN (0, 1) OR T1.TYPE = 'V')            
            AND
                T2.NAME <> 'DBO'
            GROUP BY 
                 T2.NAME
                ,T1.NAME
                ,T1.TYPE
            ;        
            """
            schema_tables_views=pd.read_sql(sql,conn)

            table_counts = schema_tables_views[schema_tables_views['OBJECT_TYPE']=='U'].groupby("SCHEMA_NAME").size().reset_index(name="TABLE_COUNT")
            view_counts = schema_tables_views[schema_tables_views['OBJECT_TYPE']=='V'].groupby("SCHEMA_NAME").size().reset_index(name="VIEW_COUNT")
            return schema_tables_views,view_counts,table_counts
        
    def plot_schemas(self):
        fig,axes = plt.subplots(1,2,figsize=(12,5))
        df=self.table_counts_df.sort_values(by='TABLE_COUNT',ascending=False)
        sns.barplot(data=df, x="TABLE_COUNT", y="SCHEMA_NAME",palette = 'viridis', edgecolor=".2",hue='SCHEMA_NAME',legend=False,ax=axes[0])
        axes[0].set_xlabel('TABLE COUNT')
        axes[0].set_ylabel('SCHEMA NAME')
        axes[0].set_title('TABLE COUNT BY SCHEMA')
        df=self.view_counts_df.sort_values(by='VIEW_COUNT',ascending=False)
        sns.barplot(data=df, x="VIEW_COUNT", y="SCHEMA_NAME",palette = 'viridis', edgecolor=".2",hue='SCHEMA_NAME',legend=False,ax=axes[1])
        axes[1].set_xlabel('VIEW COUNT')
        axes[1].set_ylabel('SCHEMA NAME')
        axes[1].set_title('VIEW COUNT BY SCHEMA')
        plt.tight_layout()
        plt.show()

    def plot_schema(self,schema_name):
        df=self.schema_tables_views_df[
            (
                (self.schema_tables_views_df['SCHEMA_NAME']==schema_name) & 
                (self.schema_tables_views_df['OBJECT_TYPE']=='U')
            )
        ].sort_values(by='ROW_COUNT',ascending=False)
        fig,ax=plt.subplots(figsize=(10,5))
        sns.barplot(data=df, x="ROW_COUNT", y="TABLE_NAME",palette = 'viridis', edgecolor=".2",hue='TABLE_NAME',legend=False)
        ax.set_xlabel('ROW COUNT')
        ax.set_ylabel('TABLE NAME')
        ax.set_title(f'ROW COUNT BY TABLE FOR SCHEMA {schema_name}')
        if df['ROW_COUNT'].max() > 10**4:
            plt.xscale('log')
        plt.show()