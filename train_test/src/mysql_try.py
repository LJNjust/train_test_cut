import mysql.connector
import pandas as pd
from  sqlalchemy import create_engine
df=pd.DataFrame([[1,'xxx'],[2,'uuu']],columns=list('ab'))

engine=create_engine('mysql+mysqldb://root:ljn@localhost:3306/pydata?charset=utf8')
pd.io.sql.to_sql(df,'pydata_go',engine,schema='pydata',if_exists='append')