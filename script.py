# Import our dependencies
from sklearn.model_selection import train_test_split
import pandas as pd

#  Import and read the charity_data.csv.
application_df = pd.read_csv("charity_data.csv")
application_df.head()

# Drop the non-beneficial ID columns, 'EIN' and 'NAME'.
# also, I'm dropping ask amount because it's numeric, and I'm dropping income amount just because it has 9 options, and that's a lot combinatorially.
# I could partition the income amount into "other", but I just didn't want to take the time
application_df=application_df.drop(['EIN', 'NAME','ASK_AMT','INCOME_AMT'], axis=1)

# Determine which values to replace if counts are less than ...?
replace_application = list(application_df['APPLICATION_TYPE'].value_counts().loc[application_df['APPLICATION_TYPE'].value_counts() < 1100].index)

# Replace in dataframe
for app in replace_application:
    application_df.APPLICATION_TYPE = application_df.APPLICATION_TYPE.replace(app,"Other")
    
# Determine which values to replace if counts are less than ..?
replace_class = list(application_df['CLASSIFICATION'].value_counts().loc[application_df['CLASSIFICATION'].value_counts() < 1900].index)

# Replace in dataframe
for cls in replace_class:
    application_df.CLASSIFICATION = application_df.CLASSIFICATION.replace(cls,"Other")




from helper_functions import predict_row, choose, do_the_split, plot_stuff
from sklearn.metrics import accuracy_score

# Split the preprocessed data into a training and testing dataset
all_train, all_test = train_test_split(
    application_df,  test_size=0.33)

depth = 2

metadata = do_the_split(all_train,0,depth,'IS_SUCCESSFUL')



# import pandas as pd
# small_df = pd.DataFrame({
#     'a':[0,0,0,0,1,1,1,1],
#     'b':[0,0,1,1,0,0,1,1],
#     'c':[0,1,0,1,0,1,0,1],
#     'y':[1,0,0,1,0,0,0,1],
# })
#metadata = do_the_split(small_df,0,4,'y')

#small_df['pred']=[choose(predict_row(small_df.iloc[i],metadata)) for i in range(len(small_df.index))]

#accuracy_score(small_df['y'],small_df['pred'])



import matplotlib.pyplot as plt

ys= all_train['IS_SUCCESSFUL'].value_counts().sort_index()
ys_in = metadata['ys_in']
ys_out = metadata['ys_out']
ys_in_in = metadata['next_split_in']['ys_in']
ys_in_out = metadata['next_split_in']['ys_out']
ys_out_in = metadata['next_split_out']['ys_in']
ys_out_out = metadata['next_split_out']['ys_out']

plt.bar(ys.index,ys)

ys_list = plot_stuff(0,depth,[],metadata)

i = 2

for ys in ys_list:
    plt.bar([i,i+1],[ys[0],ys[1]])
    i = i + 2

plt.show()
