def keep_regs(df, regs):
    """ Example function. Keep only the subset regs of regions in data.

    Args:
        df (pd.DataFrame): pandas dataframe 

    Returns:
        df (pd.DataFrame): pandas dataframe

    """ 
    
    for r in regs:
        I = df.reg.str.contains(r)
        df = df.loc[I == False] # keep everything else
    
    return df

#import pandas as pd

#data = ['alcohol','smokers','lifeexp65','socsupport']
#for i in data:
 #   data[i] = pd.read_csv(f"{i}.csv")
  #  print(i)