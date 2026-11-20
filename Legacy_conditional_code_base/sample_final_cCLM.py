import numpy as np
import torch
import time
import os
import argparse
from model_cCLM import RNN
from helper_functions import Vocabulary, generate_properties, seq_to_smiles
from tqdm import tqdm
import pandas as pd
from rdkit import Chem, RDLogger
import random
RDLogger.DisableLog('rdApp.*')


def sampler(n, model_path, output_file, batch_size, prop, scaler_file, voc_file, temperature, property_ranges, properties_column, use_range):
    voc = Vocabulary(init_from_file=voc_file)
    Agent = RNN(voc, num_props=len(properties_column))

    # Always load the trained weights. The original guarded this with
    # `if torch.cuda.is_available()`, so on a CPU-only machine it silently sampled
    # from an untrained network (and still exited 0). Identical behaviour on GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Agent.rnn.load_state_dict(torch.load(model_path, map_location=device))

    sampled_mols = set()

    with tqdm(total=n, desc="Processing SMILES", unit="smiles") as pbar:
        for i in range(1, 1000000000000000000):
            if use_range:
                if len(property_ranges) % 2 != 0:
                    raise ValueError("`--property_ranges` must contain an even number of values (LOW HIGH pairs for each property).")

                prop_range_tuples = [
                    tuple(property_ranges[i:i+2])
                    for i in range(0, len(property_ranges), 2)
                ]

                # Now pass this list of tuples
                properties1, properties2 = generate_properties(
                    sampling_batch_size=batch_size,
                    property=prop,
                    scaler=scaler_file,
                    use_random=True,
                    property_ranges=prop_range_tuples,
                    property_columns=properties_column
                )
            else:
                properties1, properties2 = generate_properties(
                    batch_size, property=prop, scaler=scaler_file,
                    use_random=False, property_ranges=None,
                    property_columns=properties_column
                )

            #print(f"Sampling for property {properties2}!")

            with torch.no_grad():
                seqs, agent_likelihood = Agent.sample(batch_size, temperature=temperature, properties=properties2)
                smiles = seq_to_smiles(seqs, voc)
                valid = 0

                for smile, likelihood in zip(
                    smiles,
                    agent_likelihood.detach().cpu().numpy()
                ):
                    mol = Chem.MolFromSmiles(smile)
                    if mol is not None:
                        valid += 1
                        prop_str = ",".join(map(str, properties1))  # Convert array to comma-separated string
   
                        sampled_mols.add((smile, likelihood, prop_str))
                        pbar.update(1)  # Update progress bar for each valid SMILES

            if i % 5 == 0:
                print(f'{(valid / batch_size) * 100:.2f}% valid SMILES')

            if len(sampled_mols) >= n:
                print('Sampling Complete!')
                break

    df = pd.DataFrame(sampled_mols, columns=['smiles', 'agent_likelihood', 'property_seed'])
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample molecules with a pretrained cCLM model.")
    parser.add_argument('--n', type=int, required=True, help='Total number of SMILES to sample.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model checkpoint.')
    parser.add_argument('--output_file', type=str, required=True, help='Output CSV file to save sampled SMILES.')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for sampling.')
    parser.add_argument('--prop', type=float, nargs='+', default=1.0, required=False, help='Property seed values (used if --use_range is False).')
    parser.add_argument('--scaler_file', type=str, required=True, help='Path to the scaler .pkl file.')
    parser.add_argument('--voc_file', type=str, default='Voc_chembl_Jan13', help='Path to the vocabulary file.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature.')
    parser.add_argument('--property_ranges', type=float, nargs='+', help='Property range values in LOW HIGH pairs.')
    parser.add_argument('--property_columns', type=str, nargs='+', required=True, help='Names of property columns.')
    parser.add_argument('--use_range', action='store_true', help='Use property range instead of fixed property values.')

    args = parser.parse_args()

    sampler(
        n=args.n,
        model_path=args.model_path,
        output_file=args.output_file,
        batch_size=args.batch_size,
        prop=args.prop,
        scaler_file=args.scaler_file,
        voc_file=args.voc_file,
        temperature=args.temperature,
        property_ranges=args.property_ranges,
        properties_column=args.property_columns,
        use_range=args.use_range
    )

