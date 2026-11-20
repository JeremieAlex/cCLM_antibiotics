
import numpy as np
import random
import re
import pickle
from rdkit import Chem
import sys
import time
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import RobustScaler
import joblib


class Vocabulary(object):
    """A class for handling encoding/decoding from SMILES to an array of indices"""
    def __init__(self, init_from_file=None, max_length=150):
        self.special_tokens = ['END', 'START'] #TODO: Add padding index to see if makes difference in training. Also implemnt in embedding layer
        self.additional_chars = set()
        self.chars = self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars)))) 
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.max_length = max_length
        if init_from_file: self.init_from_file(init_from_file)

    def encode(self, char_list):
        """Takes a list of characters (eg '[C@@H]') and encodes to array of indices"""
        smiles_matrix = np.zeros(len(char_list), dtype=np.float32)
        for i, char in enumerate(char_list):
            smiles_matrix[i] = self.vocab[char]
        return smiles_matrix

    def decode(self, matrix):
        """Takes an array of indices and returns the corresponding SMILES"""
        chars = []
        for i in matrix:
            if i == self.vocab['END']: break
            chars.append(self.reversed_vocab[i])
        smiles = "".join(chars)
        smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smiles

    def tokenize(self, smiles):
        """Takes a SMILES and return a list of characters/tokens"""
        smiles = replace_halogen(smiles)
        char_list = list(smiles)
        tokenized = []
        for char in char_list:
            if char.startswith('['):
                tokenized.append(char)
            else:
                chars = [unit for unit in char]
                [tokenized.append(unit) for unit in chars]
        tokenized.append('END')
        return tokenized

    def add_characters(self, chars):
        """Adds characters to the vocabulary"""
        for char in chars:
            self.additional_chars.add(char)
        char_list = list(self.additional_chars)
        char_list.sort()
        self.chars = char_list + self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}


    def init_from_file(self, file):
        """Takes a file containing \n separated characters to initialize the vocabulary"""
        with open(file, 'r') as f:
            chars = f.read().split()
        self.add_characters(chars)

    def __len__(self):
        return len(self.chars)

    def __str__(self):
        return "Vocabulary containing {} tokens: {}".format(len(self), self.chars)


class MolData(Dataset):
    """Custom PyTorch Dataset that takes a CSV file containing SMILES and properties."""

    def __init__(self, fname, voc, smiles_column='SMILES', property_columns=['hit']):
        """
        Args:
            fname (str): Path to the CSV file.
            voc (Vocabulary): A Vocabulary instance.
            smiles_column (str): The name of the column containing SMILES strings.
            property_columns (list): List of column names containing properties. If None, all other columns are used as properties.
        """
        self.voc = voc
        self.smiles = []
        self.properties = []

        # Load the CSV file using pandas
        df = pd.read_csv(fname)

        # Ensure the specified SMILES column exists
        if smiles_column not in df.columns:
            raise ValueError(f"SMILES column '{smiles_column}' not found in the dataset.")

        # Ensure all specified property columns exist
        for col in property_columns:
            if col not in df.columns:
                raise ValueError(f"Property column '{col}' not found in the dataset.")

        # Extract SMILES strings and properties
        self.smiles = df[smiles_column].tolist()
        self.properties = df[property_columns].values.tolist()  # Convert properties to list of lists

    def __getitem__(self, i):
        mol = self.smiles[i]
        tokenized = self.voc.tokenize(mol)
        encoded = self.voc.encode(tokenized)
        properties = torch.tensor(self.properties[i], dtype=torch.float32)
        return Variable(torch.tensor(encoded, dtype=torch.long)), Variable(properties)

    def __len__(self):
        return len(self.smiles)

    @classmethod
    def collate_fn(cls, arr):
        """Function to take a list of encoded sequences and turn them into a batch."""
        sequences, properties = zip(*arr)
        max_length = max([seq.size(0) for seq in sequences])
        collated_sequences = torch.zeros(len(sequences), max_length, dtype=torch.long)
        for i, seq in enumerate(sequences):
            collated_sequences[i, :seq.size(0)] = seq
        collated_properties = torch.stack(properties)  # Stack properties into a batch tensor
        return collated_sequences, collated_properties

def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)

    return string

def tokenize(smiles):
    """Takes a SMILES string and returns a list of tokens.
    This will swap 'Cl' and 'Br' to 'L' and 'R' and treat
    '[C@H]' as one token."""
    smiles = replace_halogen(smiles)
    char_list = list(smiles)
    tokenized = []
    for char in char_list:
        if char.startswith('['):
            tokenized.append(char)
        else:
            chars = [unit for unit in char]
            [tokenized.append(unit) for unit in chars]
    tokenized.append('END')
    return tokenized



def seq_to_smiles(seqs, voc):
    """Takes an output sequence from the RNN and returns the
       corresponding SMILES."""
    smiles = []
    for seq in seqs.cpu().numpy():
        smiles.append(voc.decode(seq))
    return smiles


def canonicalize_smiles_from_file(fname, smiles_column='SMILES', output_csv='canonicalized_smiles.csv'):
    """Reads a CSV file with a SMILES column, returns a list of canonical isomeric SMILES, 
       and writes the updated CSV file with only valid rows and a new 'Canonical_SMILES' column.

    Args:
        fname (str): Path to the input CSV file.
        smiles_column (str): Name of the column containing SMILES strings.
        output_csv (str): Path to save the updated CSV file with canonical SMILES.

    Returns:
        list: A list of canonical SMILES strings.
    """
    df = pd.read_csv(fname)

    # Ensure the specified SMILES column exists
    if smiles_column not in df.columns:
        raise ValueError(f"SMILES column '{smiles_column}' not found in the dataset.")

    # Lists to store valid canonical SMILES and rows to keep
    canonical_smiles_list = []
    rows_to_keep = []

    for i, smiles in enumerate(df[smiles_column]):
        if i % 10000 == 0:
            print(f"{i} lines processed.")
        
        mol = Chem.MolFromSmiles(smiles)
        if mol and filter_mol(mol):
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            canonical_smiles_list.append(canonical_smiles)
            rows_to_keep.append(i)  # Keep track of the row indices for valid SMILES

    # Filter the DataFrame to keep only valid rows
    updated_df = df.iloc[rows_to_keep].copy()
    updated_df['SMILES'] = canonical_smiles_list

    # Save the updated DataFrame to a new CSV file
    updated_df.to_csv(output_csv, index=False)
    print("=======================Scaled and Canocalized Data======================")
    print("updated df:", updated_df)
    print("len of updated df:", len(updated_df))
    print(f"Updated CSV with canonical SMILES saved to {output_csv}")

    # Return the list of canonical SMILES and the updated DataFrame
    return canonical_smiles_list, updated_df

def filter_mol(mol, max_heavy_atoms=100, min_heavy_atoms=10, element_list=[6,7,8,9,16,17,35]):
    """Filters molecules on number of heavy atoms and atom types"""
    if mol is not None:
        num_heavy = min_heavy_atoms<mol.GetNumHeavyAtoms()<max_heavy_atoms
        elements = all([atom.GetAtomicNum() in element_list for atom in mol.GetAtoms()])
        if num_heavy and elements:
            return True
        else:
            return False


def filter_on_chars(smiles_list, chars):
    """Filters SMILES on the characters they contain.
       Used to remove SMILES containing very rare/undesirable
       characters."""
    smiles_list_valid = []
    for smiles in tqdm(smiles_list):
        tokenized = tokenize(smiles)
#         print(tokenized)
        if all([char in chars for char in tokenized][:-1]):
            smiles_list_valid.append(smiles)
    print('Filtered library size: %d'%len(smiles_list_valid))
    return smiles_list_valid


def filter_file_on_chars(smiles_list, voc_fname):
    """Filters a SMILES file using a vocabulary file.
       Only SMILES containing nothing but the characters
       in the vocabulary will be retained."""
    smiles = []
    chars = []
    for line in smiles_list:
        smiles.append(line.split()[0])
    with open(voc_fname, 'r') as f:
        for line in f:
            chars.append(line.split()[0])
    print('Vocabulary size: %d'%len(chars))
    print('Origin library size: %d'%len(smiles))
    valid_smiles = filter_on_chars(smiles, chars)
    with open("tl_filtered.smi", 'w') as f:
        for smiles in valid_smiles:
            f.write(smiles + "\n")

def construct_vocabulary(smiles_list,fname):
    """Returns all the characters present in a SMILES file.
       Uses regex to find characters/tokens of the format '[x]'."""
    add_chars = set()
    for i, smiles in enumerate(smiles_list):
        smiles = replace_halogen(smiles)
        char_list = list(smiles)
        for char in char_list:
            if char.startswith('['):
                add_chars.add(char)

            else:
                chars = [unit for unit in char]
                [add_chars.add(unit) for unit in chars]

    print("Number of characters: {}".format(len(add_chars)))
    with open(fname, 'w') as f:
        for char in add_chars:
            f.write(char + "\n")
    return add_chars

def Variable(tensor):
    """Wrapper for torch.autograd.Variable that also accepts
       numpy arrays directly and automatically assigns it to
       the GPU. Be aware in case some operations are better
       left to the CPU."""
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)



def scale_and_plot_properties(input_csv, output_csv, scaler_filename, smiles_column='SMILES', property_columns=['hit'], premade_scaler=None):
    # Load the data from the CSV file
    df = pd.read_csv(input_csv)

    # Ensure the specified SMILES column exists
    if smiles_column not in df.columns:
        raise ValueError(f"SMILES column '{smiles_column}' not found in the dataset.")

    # Ensure all specified property columns exist
    for col in property_columns:
        if col not in df.columns:
            raise ValueError(f"Property column '{col}' not found in the dataset.")    

   
    if premade_scaler:
        scaler = joblib.load(premade_scaler)
        scaler_filename = premade_scaler
        df[property_columns] = scaler.transform(df[property_columns])
    else: 
        scaler = RobustScaler()
        df[property_columns] = scaler.fit_transform(df[property_columns])
        joblib.dump(scaler, scaler_filename)

    # Save the scaled data to a new CSV file
    df.to_csv(output_csv, index=False)
    print("=======================Scaled Data======================")
    print(df)


    # Create a PDF file to save the plots
    with PdfPages('property_check.pdf') as pdf:
        for column in property_columns:
            plt.figure(figsize=(12, 6))

            # Plot histogram
            plt.subplot(1, 2, 1)
            df[column].hist(bins=30, edgecolor='black')
            plt.title(f'Histogram of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')

            # Plot boxplot
            plt.subplot(1, 2, 2)
            df.boxplot(column=column)
            plt.title(f'Boxplot_of_{column}')
            pdf.savefig()  # Save the current figure to the PDF
            plt.close()  # Close the figure to free memory

    return output_csv, scaler_filename


def parse_property_range(range_str):
    # Convert the input string to a list of tuples
    try:
        return [tuple(map(float, item.split(','))) for item in range_str.split(';')]
    except ValueError:
        raise argparse.ArgumentTypeError("Property range must be a list of tuples in the format '0.2,1;0.3,0.8'")
    


def filter_file_on_chars_csv(df, voc_fname, output_csv):
    """Filters a DataFrame using a vocabulary file.
       Only rows with SMILES containing nothing but the characters
       in the vocabulary will be retained."""
    smiles = df['SMILES'].tolist()
    smiles_list = []
    chars = []
    
    for line in tqdm(smiles):
        smiles_list.append(line.split()[0])
    with open(voc_fname, 'r') as f:
        for line in f:
            chars.append(line.split()[0])
    print('Vocabulary size: %d' % len(chars))
    print('Origin library size: %d' % len(smiles))
    valid_smiles = filter_on_chars(smiles, chars)
    
    # Filter the DataFrame based on valid SMILES
    filtered_df = df[df['SMILES'].isin(valid_smiles)]
    print('Filtered DataFrame:', filtered_df)
    
    # Save the filtered DataFrame to a CSV file
    filtered_df.to_csv(output_csv, index=False)
    print(f"Filtered data saved to {output_csv}")


class EarlyStopping():
    """
    Monitor the training process to stop training early if the model shows
    evidence of beginning to overfit the validation dataset. Does not save
    the model; train_cCLM.py saves the final model when training ends.

    Args:
        patience: (int) if the validation loss fails to improve for this
                  number of consecutive steps, training will be stopped.
        threshold: (float) the minimum decrease in the validation loss to reset the counter.
    """

    def __init__(self, patience=2, threshold=0.5):
        self.patience = patience
        self.threshold = threshold
        self.counter = 0
        self.best_loss = None
        self.step_at_best = 0
        self.stop = False
        self.loss_history = []  # List to store validation loss values
        print(f"Instantiated early stopping with patience={self.patience} and threshold={self.threshold}")

    def __call__(self, val_loss, model, output_file, step_idx):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.step_at_best = step_idx
        elif val_loss >= self.best_loss - self.threshold:
            # loss is not decreasing significantly
            self.counter += 1
            print(f"Early stopping counter: {self.counter}", f'Best loss: {self.best_loss}', f'Current loss: {val_loss}')
            if self.counter >= self.patience:
                self.stop = True
                print("Stopping early with best loss: " + str(self.best_loss))
        else:
            # loss is decreasing significantly
            self.best_loss = val_loss
            self.step_at_best = step_idx
            self.counter = 0
            print(f"Resetting early stopping counter. Best loss: {self.best_loss}")

        self.loss_history.append(val_loss)  # Append validation loss to the history list

    def save_model(self, model, output_file):
        torch.save(model.state_dict(), output_file)
        
    def save_loss_history(self, output_file):
        with open(output_file, 'w') as f:
            for loss in self.loss_history:
                f.write(str(loss) + '\n')


def scale_value(column_name, value, property_columns, scaler):
    
    if column_name not in property_columns:
        raise ValueError(f"Column {column_name} is not in the list of property columns.")
    
    value_df = pd.DataFrame([[value] * len(property_columns)], columns=property_columns)
    scaled_value = scaler.transform(value_df)[0][property_columns.index(column_name)]
    return scaled_value



def generate_properties(
    sampling_batch_size,
    scaler,
    use_random=False,
    property_ranges=[(0.8, 1.0)],
    property=1,
    property_columns=['binary_activity']
):


    # Load scaler
    scaler = joblib.load(scaler)

    # Generate random property values
    if use_random:
        # property_ranges must be a list of tuples [(low, high), ...]
        properties1 = [random.uniform(low, high) for (low, high) in property_ranges]
    else:
        # property is a list of property values, length must match property_columns
        if isinstance(property, list):
            properties1 = property  # Already a list, use directly
        else:
            properties1 = [property] #change this to the property value you want to use

    # Scale the properties individually
    scaled_properties1 = [
        scale_value(property_columns[i], properties1[i], property_columns, scaler)
        for i in range(len(properties1))
    ]

    # Make a batch tensor of shape (sampling_batch_size, num_props)
    properties2 = torch.tensor(
        [scaled_properties1] * sampling_batch_size,
        dtype=torch.float32,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Properties generated: {properties1}")
    print(f"Properties scaled: {scaled_properties1}")

    return properties1, properties2

