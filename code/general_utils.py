'''
Common utils code
'''
import torch 
import numpy as np

PRINT_LINEWIDTH = 80 # cmd/notebook line width for pretty log info print

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu') # used for AEC (for maximum compatibility)

# print centered title with decoration char
def print_title(text, total_width=PRINT_LINEWIDTH, decoration_char="+", head_line=True, tail_line=False, newline=True):
    text_len = len(text) + 2 # 2 space
    available_space = total_width - text_len
    left_padding = available_space // 2  
    right_padding = available_space - left_padding 
    content = ''
    if newline: content += '\n'
    if head_line: content += (decoration_char * total_width + '\n')
    content += f"{decoration_char * left_padding} {text} {decoration_char * right_padding}"
    if tail_line: content += ('\n'+decoration_char * total_width)
    print(content)

# concat X_in, X_out into Tabular (used for DeepLog case) 
# NOTE: np.ndarry -> np.ndarry
def seq2tab(X_in, X_out):
    assert len(X_in) == len(X_out)
    X_tab = np.concatenate((X_in, np.reshape(X_out, (-1, 1))), axis=1)
    return X_tab

# split Tabular into X_in (seq) and X_out (label)
# NOTE: np.ndarry -> np.ndarry
def tab2seq(X_tab, window_size):
    X_in = X_tab[:,:window_size]
    X_out = np.reshape(X_tab[:, window_size], (-1))
    return X_in, X_out


# used with GEAD.clip_outlier_nnprob (see this for more)
# NOTE: mostly used for Kistune case
def clip_outlier_nnprob_func(nn_prob, ad_thres, weight, replace=False):
    if not replace:
        nn_prob = np.copy(nn_prob)
    nn_prob[nn_prob>ad_thres*weight] = ad_thres*weight
    return nn_prob

def merge_highlight(*highlight_dicts):
    merged_highlight = {}
    for highlight in highlight_dicts:
        merged_highlight.update(highlight)
    return merged_highlight


# load packaged dataset and show basic info
def load_dataset(dataset_name, dataset_pth, verbose=False):
    try:
        dataset = np.load(dataset_pth, allow_pickle=True)
        if verbose:
            y_iid, y_ood = dataset['y_iid'], dataset['y_ood']
            print(f"IID data: total {len(y_iid)} items (ben: {len(y_iid[y_iid==0])}, mal: {len(y_iid[y_iid==1])})")
            print(f"OOD data: total {len(y_ood)} items (ben: {len(y_ood[y_ood==0])}, mal: {len(y_ood[y_ood==1])})")
        print(f'Successfully load [{dataset_name}] dataset!')
        return dataset
    except Exception as e:
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {e}")
        print("Note: dataset must follow pre-defined .npz format!")
    