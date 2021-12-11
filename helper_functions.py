import numpy as np

from itertools import chain, combinations

from scipy.stats import entropy

def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    for z in list(powerset([1,2,3,4,5,6,7,8,9])):
        for x in z:
            print(x)
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

assert (1,2) in powerset([1,2,3])




def predict_row(row, model_dict):

    if row[model_dict['column']] in model_dict['combo']:
        try:
            return predict_row(row,model_dict['next_split_in'])
        except:
            return model_dict['ys_in']
    else:
        try:
            return predict_row(row,model_dict['next_split_out'])
        except:
            return model_dict['ys_out']

import pandas as pd
test_df_row = pd.DataFrame([{
    'f1':3,
    'f2':2,
    'y':1
}]).loc[0]
test_model_dict = {
    'column':'f1',
    'combo':(3,),
    'ys_in':{1:1},
    'ys_out':None,
}

assert  predict_row(test_df_row,test_model_dict) == {1:1}

del test_df_row, test_model_dict, pd



def choose(dict):
    """
    This function is very important in the theory.
    This is important because when given a distribution of options, and you're trying to guess
    the correct one, your best strategy is always to pick the one with highest probability.

    It's also important because when predicting a binary variable (or any categorical variable, for that matter),
    each time you make an entropy-reducing split, it does not necessarily mean that either side of the split needs to have a different guess.
    For example, let's say you have the following points:
    (0,0)
    (1,0)
    (2,0)
    (3,1)
    (4,0)
    (5,0)
    Your initial prediction is just to guess zeros, and your entropy is -5/6 * log2(5/6) - 1/6 * log2(1/6) = 0.6500
    And then the entropy-reducing split would be at 2.5, so you'd have: 
    (0,0)
    (1,0)
    (2,0)
    -------
    (3,1)
    (4,0)
    (5,0)
    and on either side of the split, you'd still predict 0 as your best guess. Your new entropy is:
    1/2 * ( 0 ) + 1/2 * ( -1/3 * log2(1/3) - 2/3 * log2(2/3) ) = 0.4591
    Thus, entropy has been reduced, but you still make the same predictions -- the benefit is that you're more confident about the guess.


    I found this implementation here: https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
    """
    return max(dict, key=dict.get)

import pandas as pd
assert choose({1:4,7:10}) == 7
del pd



def get_column_data(df,column,y_col):

    ys_orig_counts_dict =  dict(df[y_col].value_counts())

    num_of_rows_total = len(df.index)

    assert num_of_rows_total == sum(ys_orig_counts_dict.values())

    entropy_orig = entropy(
        [y / num_of_rows_total for y in ys_orig_counts_dict.values()],
        base=2
    )

    column_vals = df[column].unique()

    split_metadata_dict = {}

    for combo in powerset(column_vals):

        if len(combo) > len(column_vals) / 2:

            # only consider combos that are less than half the length of "val"
            pass

        else:

            # filter to the rows that are in and out
            df_in = df[df[column].isin(combo)]
            df_out = df[~df[column].isin(combo)]
            
            # get the counts of the y values both in and out
            # for example, it would be [0: 52, 1: 35]
            ys_in_counts_dict = dict(df_in[y_col].value_counts())
            ys_out_counts_dict = dict(df_out[y_col].value_counts())
            
            num_of_rows_in = len(df_in.index) # sum(ys_in_counts_dict.values())
            num_of_rows_out = len(df_out.index) # sum(ys_out_counts_dict.values())

            # calculate the total number of rows in and r
            assert num_of_rows_in == sum(ys_in_counts_dict.values())
            assert num_of_rows_out == sum(ys_out_counts_dict.values())

            new_entropy = \
                entropy(
                    [y / num_of_rows_in for y in ys_in_counts_dict.values()],
                    base=2
                ) * num_of_rows_in / num_of_rows_total + \
                entropy(
                    [y / num_of_rows_out for y in ys_out_counts_dict.values()],
                    base=2
                ) * num_of_rows_out / num_of_rows_total

            entropy_pct_reduction = 100 * (entropy_orig - new_entropy)/entropy_orig

            drop = {
                'column':column,
                'combo':combo,
                'entropy_pct_reduction':entropy_pct_reduction,
                'ys_in':ys_in_counts_dict,
                'ys_out':ys_out_counts_dict
            } 

            split_metadata_dict[combo] = drop

    return split_metadata_dict

import pandas as pd
test_df = pd.DataFrame({
    'x':[0,1,0,1],
    'y':[1,0,1,0]
})
col = 'x'
col_data_dict = get_column_data(test_df,col,'y')
split_0_dict = col_data_dict[(0,)]
split_1_dict = col_data_dict[(1,)]
assert split_0_dict['column'] == col
assert split_0_dict['entropy_pct_reduction'] == 100
assert (
    split_0_dict['ys_in'] == {1:2} and
    split_0_dict['ys_out'] == {0:2}
)
(
    split_1_dict['ys_in'] == {0:2} and
    split_1_dict['ys_out'] == {1:2}
)
del pd, test_df, col, col_data_dict, split_0_dict,split_1_dict


def generate_split_metadata(df,y_col):

    cols_data_dict = {}

    for col in df.drop(y_col,axis=1).columns:
        cols_data_dict[col] = get_column_data(df,col,y_col)

    potential_splits = []

    for col in cols_data_dict.keys():
        for tuple_key in cols_data_dict[col].keys():
            potential_splits.append({
                "column":col,
                "tuple_key": tuple_key,
                'data':cols_data_dict[col][tuple_key]
            })
    
    list_of_entropy_drops_unscaled = [potential_split['data']['entropy_pct_reduction'] for potential_split in potential_splits]

    sum_of_list_of_entropy_drops = sum(list_of_entropy_drops_unscaled)

    list_of_entropy_drops = [x/sum_of_list_of_entropy_drops for x in list_of_entropy_drops_unscaled]

    from random import choices

    # returning a dict with the distribution, etc
    return choices(potential_splits,list_of_entropy_drops)[0]['data']


import pandas as pd
test_df = pd.DataFrame({
    'x':[0,1,0,1],
    'y':[1,0,1,0]
})
col_data_dict = generate_split_metadata(test_df,'y')
assert col_data_dict['column'] == 'x'
assert col_data_dict['entropy_pct_reduction'] == 100
if col_data_dict['combo'] == (0,):
    assert (
        col_data_dict['ys_in'] == {1:2} and
        col_data_dict['ys_out'] == {0:2}
    )
else:
    assert (
        col_data_dict['ys_in'] == {0:2} and
        col_data_dict['ys_out'] == {1:2}
    )
del pd, test_df, col_data_dict


def do_the_split(df,num_of_splits,max_num,y_col):

    metadata = generate_split_metadata(df,y_col)

    num_of_splits = num_of_splits + 1

    if num_of_splits == max_num:

        return metadata

    else:
    
        # here we're using the split metadata to make the split
        filter_list  = df[metadata['column']].isin(metadata['combo'])

        df_in = df[filter_list]
        df_out = df[~filter_list]

        metadata['next_split_in'] = do_the_split(df_in,num_of_splits,max_num,y_col)
        metadata['next_split_out'] = do_the_split(df_out,num_of_splits,max_num,y_col)

        return metadata





def plot_stuff(depth, max_depth, y_list, meta_data):
    depth = depth + 1
    if depth == max_depth:
        y_list.append(meta_data['ys_in'])
        y_list.append(meta_data['ys_out'])
    else:
        y_list.append(plot_stuff(depth,max_depth,y_list,meta_data['next_split_in']))
        y_list.append(plot_stuff(depth,max_depth,y_list,meta_data['next_split_out']))
    return y_list



test_metadata = {'column': 'AFFILIATION',
 'combo': ('Independent', 'Family/Parent', 'Other'),
 'entropy_pct_reduction': 9.873902127068506,
 'ys_in': {1: 8677, 0: 3734},
 'ys_out': {0: 7048, 1: 3521},
 'next_split_in': {'column': 'ORGANIZATION',
  'combo': ('Association', 'Co-operative'),
  'entropy_pct_reduction': 0.827664779101066,
  'ys_in': {1: 842, 0: 635},
  'ys_out': {1: 7835, 0: 3099}},
 'next_split_out': {'column': 'APPLICATION_TYPE',
  'combo': ('T5', 'T6'),
  'entropy_pct_reduction': 6.605084236713802,
  'ys_in': {1: 747, 0: 212},
  'ys_out': {0: 6836, 1: 2774}}}

solution_val = [
    {1: 842, 0: 635},
    {1: 7835, 0: 3099},
    {1: 747, 0: 212},
    {0: 6836, 1: 2774}
]
print(plot_stuff(0,2,[],test_metadata))
#assert plot_stuff(0,2,[],test_metadata) == solution_val