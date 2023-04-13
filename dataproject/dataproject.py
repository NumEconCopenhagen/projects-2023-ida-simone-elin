
#Cleaning the 4 datasets simultaneously
def dataclean(df,value):
   """function that drops and renames columns"""
   drop_these = ['INDICATOR','MEASURE','FREQUENCY','Flag Codes'] 
   df.drop(drop_these, axis=1, inplace=True)
   df.rename(columns={'LOCATION':'country', 'TIME':'year', 'Value':value}, inplace=True)   
   return df





