"""
module database_descriptor.py

This module provides functionality for generating a quick overview of the schemas, 
tables, views and relations between tables in a database.

Author: Andreas Rasmusson
Date: December 13, 2024

Usage:
    connection_string = [your connection string here]
    d = DatabaseDescriptor(connection_string)
    d.plot_schemas
    d.plot_schema([your chosen schema_1 here])
    .
    .
    .
    d.plot_schema([your chosen schema_k here])

Classes:

    DatabaseDescriptor - Provides all the functionality - connection to database, needed queries and plotting functions

"""

# Perform necessary imports
from sqlalchemy import create_engine
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import networkx as nx

# Set seaborn theme
sns.set_theme(style="darkgrid")  

# A query for getting the schema names, table names and
# table row counts for those schemas that would be
# interesting for a normal user.
#
# In this query, we use the system views SYS.OBJECTS, 
# SYS.SCHEMAS and SYS.DM_DB_PARTITION_STATS which, among
# other things contain information about:
# 
# 1. SYS.OBJECTS - Information such as object type (table, view, etc), creation and modifcation times
# 2. SYS.SCHEMAS - schema names
# 3. SYS.DM_DB_PARTITION_STATS - Information about row counts for tables
sql_schemas="""
SELECT 
    T2.NAME SCHEMA_NAME
    ,T1.NAME TABLE_NAME
    ,TRIM(T1.TYPE) AS OBJECT_TYPE
    ,SUM(T3.ROW_COUNT) AS ROW_COUNT
FROM 
    SYS.OBJECTS T1
INNER JOIN -- Inner join here, since we are only interested in
           -- objects that belong to schemas. 
    SYS.SCHEMAS T2 
ON 
    T1.SCHEMA_ID = T2.SCHEMA_ID
LEFT JOIN -- Left join here, since we don't want to exclude views
    SYS.DM_DB_PARTITION_STATS T3 
ON 
    T1.OBJECT_ID = T3.OBJECT_ID
WHERE 
    T1.TYPE IN ('U', 'V') -- We are only interested in user tables and views
AND 
    (T3.INDEX_ID IN (0, 1) OR T1.TYPE = 'V') -- We are only interested in row counts in the actual physical storage of
                                             -- data. Excluding other indexes achieves this.
AND
    T2.NAME <> 'DBO' 
GROUP BY 
        T2.NAME
    ,T1.NAME
    ,T1.TYPE
;        
"""

# A CTE query for getting the foreign key relations between tables in the schemas
# that are relevant for a normal user. 
#
# In this query, we use the following system views:
#
# 1. SYS.TABLES - Information about tables such as name, type, creation time etc
# 2. SYS.SCHEMAS - Schema names
# 3. SYS.FOREIGN_KEYS - Information such as the name of the key, which table it 
#    points to, creation date etc
# 4. SYS.FOREIGN_KEY_COLUMNS - Information about columns associated with the 
#                              foreign key constraint
# 5. SYS.COLUMNS - Information about columns in general such as name, data type, 
#                  precision etc
#
sql_relations="""
/*
    Step 1 - get information about relevant tables
*/
WITH TABS AS (
SELECT
    T2.NAME SCHEMA_NAME
    ,T1.NAME TABLE_NAME
    ,T1.OBJECT_ID
FROM 
    SYS.TABLES T1
INNER JOIN
    SYS.SCHEMAS T2
ON
    T1.SCHEMA_ID = T2.SCHEMA_ID
WHERE -- Exclude non interesting schemas
    T2.NAME NOT IN (
        'sys', 
        'INFORMATION_SCHEMA', 
        'dbo', 
        'guest', 
        'db_accessadmin', 
        'db_backupoperator', 
        'db_datareader', 
        'db_datawriter', 
        'db_ddladmin', 
        'db_denydatareader', 
        'db_denydatawriter', 
        'db_owner', 
        'db_securityadmin'
    )
/*
    Step 2 - get information about foreign key relatations.
             It's inner joins all the way here since we want
             only mathces.
*/
), FORKEYS AS(
SELECT
    T7.NAME SCHEMA_NAME
    ,T5.NAME TABLE_NAME
    ,T6.NAME COLUMN_NAME
    ,T8.NAME IS_REFERENCED_BY_SCHEMA_NAME
    ,T3.NAME IS_REFERENCED_BY_TABLE_NAME
    ,T4.NAME IS_REFERENCED_BY_COLUMN_NAME
    ,T1.NAME VIA_FOREIGN_KEY_NAME

FROM
    SYS.FOREIGN_KEYS T1
INNER JOIN
    SYS.FOREIGN_KEY_COLUMNS T2 
ON 
    T1.OBJECT_ID = T2.CONSTRAINT_OBJECT_ID
INNER JOIN
    SYS.TABLES T3
ON 
    T2.PARENT_OBJECT_ID = T3.OBJECT_ID
INNER JOIN
    SYS.COLUMNS T4
ON 
    T2.PARENT_OBJECT_ID = T4.OBJECT_ID AND T2.PARENT_COLUMN_ID = T4.COLUMN_ID
INNER JOIN
    SYS.TABLES T5
ON 
    T2.REFERENCED_OBJECT_ID = T5.OBJECT_ID
INNER JOIN
    SYS.COLUMNS T6
ON 
    T2.REFERENCED_OBJECT_ID = T6.OBJECT_ID AND T2.REFERENCED_COLUMN_ID = T6.COLUMN_ID
INNER JOIN
    SYS.SCHEMAS T7
ON 
    T5.SCHEMA_ID = T7.SCHEMA_ID
INNER JOIN
    SYS.SCHEMAS T8
ON
    T3.SCHEMA_ID = T8.SCHEMA_ID
)
/*
    Final step - Gather the needed data from the temp tables.
                 We use a left join here since we want to keep
                 those tables in the schema that do not have
                 any relationships with other tables.
*/
SELECT
    T1.SCHEMA_NAME
    ,T1.TABLE_NAME
    ,T2.COLUMN_NAME
    ,T2.IS_REFERENCED_BY_SCHEMA_NAME
    ,T2.IS_REFERENCED_BY_TABLE_NAME
    ,T2.IS_REFERENCED_BY_COLUMN_NAME
    ,T2.VIA_FOREIGN_KEY_NAME
FROM
    TABS T1
LEFT JOIN -- Left join here since we want to keep
          -- tables that have no relationsships
    FORKEYS T2
ON
    T1.SCHEMA_NAME = T2.SCHEMA_NAME
AND
    T1.TABLE_NAME = T2.TABLE_NAME
ORDER BY
    T1.SCHEMA_NAME, CASE WHEN COLUMN_NAME IS NULL THEN 0 ELSE 1 END, TABLE_NAME
;
"""


class DatabaseDescriptor:
    """
    A class for getting information about a database.

    This class provides funktionality for generating a quick overview of the schemas, 
    tables, views and relations between tables in a database.

    Attributes:
        engine:                     sqlalchemy.engine.base.Engine
        schema_tables_views_df:     pandas.core.frame.DataFrame
        view_counts_df:             pandas.core.frame.DataFrame
        table_counts_df             pandas.core.frame.DataFrame
        relations_df:               pandas.core.frame.DataFrame

    Methods:
        plot_schemas:                                                Display two bar charts: Table counts by
                                                                                             schema and view
                                                                                             counts by schema
        plot_schema(schema_name):                                    Display a bar chart: Row counts by table
        
        plot_schema_relations(schema_name): Display a network chart: Foreign key relations between tables
    """
    def __init__(self,connection_string: str):
        # Set up the connection to the database, perform queries and load 
        # results into dataframes.     
        self.engine = create_engine(connection_string)
        self.schema_tables_views_df, self.view_counts_df, self.table_counts_df,self.relations_df = self.__get_schema_table_view_dfs__()
        
    def plot_schemas(self):
        """
        Display two bar charts: table count by schema and view count by schema
        """
        fig,axes = plt.subplots(1,2,figsize=(12,5))
        # Sort the table_counts dataframe to get the largest tables
        # at the top in the bar chart
        df=self.table_counts_df.sort_values(by='TABLE_COUNT',ascending=False)
        # Create the bar chart for the tables
        sns.barplot(
            data=df,
            x="TABLE_COUNT", 
            y="SCHEMA_NAME",
            palette = 'viridis', # Prettyfication
            edgecolor=".2",      # Prettyfication
            hue='SCHEMA_NAME',   # Prettyfication
            legend=False,
            ax=axes[0]
        )
        axes[0].set_xlabel('TABLE COUNT')
        axes[0].set_ylabel('SCHEMA NAME')
        axes[0].set_title('TABLE COUNT BY SCHEMA')
        # Sort the view_counts dataframe to get the largest views
        # at the top in the bar chart
        df=self.view_counts_df.sort_values(by='VIEW_COUNT',ascending=False)
        # Create the bar chart for the views
        sns.barplot(
            data=df, 
            x="VIEW_COUNT", 
            y="SCHEMA_NAME",
            palette = 'viridis', # Prettyfication 
            edgecolor=".2",      # Prettyfication
            hue='SCHEMA_NAME',   # Prettyfication
            legend=False,
            ax=axes[1]
        )
        axes[1].set_xlabel('VIEW COUNT')
        axes[1].set_ylabel('SCHEMA NAME')
        axes[1].set_title('VIEW COUNT BY SCHEMA')
        # Make sure nothing falls outside the figure
        plt.tight_layout()
        plt.show()

    def plot_schema(self,schema_name: str):
        """
        Display a bar chart: row count by table name

        Arguments:
            schema_name: str
        """
        # Filter the schema_tables_views_df so that we only concern
        # ourselves with tables from the schema given in the argument
        # Also, sort the result to get the tables with the largest
        # row counts on top in the bar chart.
        df=self.schema_tables_views_df[
            (
                (self.schema_tables_views_df['SCHEMA_NAME']==schema_name) & 
                (self.schema_tables_views_df['OBJECT_TYPE']=='U')
            )
        ].sort_values(by='ROW_COUNT',ascending=False)

        fig,ax=plt.subplots(figsize=(10,5))
        # Create the bar chart
        sns.barplot(
            data=df, 
            x="ROW_COUNT", 
            y="TABLE_NAME",
            palette = 'viridis', # Prettyfication
            edgecolor=".2",      # Prettyfication
            hue='TABLE_NAME',    # Prettyfication
            legend=False
        )
        ax.set_xlabel('ROW COUNT')
        ax.set_ylabel('TABLE NAME')
        ax.set_title(f'ROW COUNT BY TABLE FOR SCHEMA {schema_name}')
        # If the table counts are sufficiently large, it's better to have
        # logarithmic scaling on the row count axis
        if df['ROW_COUNT'].max() > 10**4:
            plt.xscale('log')
        
        plt.show()
    
    def plot_schema_relations(self,schema_name):
        """
        Display a network graph of the foreign key relationships
        for tables in a schema.

        Arguments:
            schema_name: str
        """
        df = self.relations_df[self.relations_df['SCHEMA_NAME'] == schema_name]
        G,pos,node_color_list = self.__build_network_graph__(df,schema_name)
        
        plt.figure(figsize=(25, 8))
        nx.draw(
            G, 
            pos, 
            with_labels=True, 
            node_size=4000, 
            node_color=node_color_list, 
            font_size=12, 
            font_weight='bold', 
            edge_color='gray', 
            linewidths=2, 
            arrowsize=25, 
            arrowstyle='-|>'
        )

        plt.title(f"TABLE RELATIONS FOR SCHEMA {schema_name}", fontsize=18, fontweight='bold')
        plt.show()

    def __build_network_graph__(self, df: pd.core.frame.DataFrame,schema_name: str) -> tuple:
        # Initialize a dictionary that will hold information about
        # the color of teach node in the graph
        node_colors = {}
        # Initialize a graph
        G = nx.DiGraph()
        # Create the nodes/edges of the graph and determine node color
        for idx, row in df.iterrows():
            table_name = row['TABLE_NAME']
            G.add_node(table_name)
            node_colors[table_name]='skyblue'
            referenced_table_name = row['IS_REFERENCED_BY_TABLE_NAME']
            # It may be the case that the current table has no relationsships
            # with other tables, so we must check this before proceeding.
            if pd.notna(referenced_table_name):
                G.add_node(referenced_table_name)
                # It may be the case that the current referenced table
                # doesn't belong to the schema we are interested in.
                # If this is so, we want to reflect this fact by having
                # the node taking a different color.
                if row['IS_REFERENCED_BY_SCHEMA_NAME'] != schema_name:
                    node_colors[referenced_table_name] = 'red'
                else:
                    node_colors[referenced_table_name] = 'skyblue'
                G.add_edge(referenced_table_name,table_name)
        
        # Position the nodes and edges (for drawing purposes)
        # We wan't a hierarchical layout for the graph. For 
        # this, graphviz needs to be installed, which may not
        # be the case. We'll try to use it and if it doesn't
        # work we'll revert back to the default layout
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
            pos = {node: (x, -y) for node, (x, y) in pos.items()}
        except ImportError:
            pos = nx.spring_layout(G)
        # Make sure the color information is in a type suitable
        # for nx.draw
        node_color_list = [node_colors[node] for node in G.nodes]
        
        return G,pos,node_color_list

        
    def __get_schema_table_view_dfs__(self) -> tuple:
        # Connect to the database, execute the queries
        # and load information into dataframes
        with self.engine.connect() as conn:            
            schema_tables_views=pd.read_sql(sql_schemas,conn)
            table_counts = schema_tables_views[schema_tables_views['OBJECT_TYPE']=='U'].groupby("SCHEMA_NAME").size().reset_index(name="TABLE_COUNT")
            view_counts = schema_tables_views[schema_tables_views['OBJECT_TYPE']=='V'].groupby("SCHEMA_NAME").size().reset_index(name="VIEW_COUNT")            
            relations = pd.read_sql(sql_relations,conn)
            return schema_tables_views,view_counts,table_counts,relations