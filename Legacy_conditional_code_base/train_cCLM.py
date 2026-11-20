#!/usr/bin/env python

import torch
from torch.utils.data import DataLoader, random_split
import pickle
from rdkit import Chem
from tqdm import tqdm
import helper_functions as ds
from helper_functions import MolData, Vocabulary, EarlyStopping, generate_properties
from model_cCLM import RNN
import pandas as pd
import argparse
import os


def validate(model, val_loader):
    model.rnn.eval()
    val_loss = 0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            sequences, properties = batch
            seqs = sequences.long()

            loss = model.likelihood(seqs, properties)
            loss = loss.mean()

            val_loss += loss.item()
    return val_loss / len(val_loader)


def pretrain(input_csv, scaler, property_range, updated_model, voc_f, property_columns, batch_size=128, learning_rate=0.0001):
    voc = Vocabulary(init_from_file=voc_f)
    moldata = MolData(input_csv, voc, property_columns=property_columns)

    total_size = len(moldata)
    train_size = int(0.85 * total_size)
    val_size = total_size - train_size
    train_data, val_data = random_split(moldata, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False,
                              collate_fn=MolData.collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, drop_last=False,
                            collate_fn=MolData.collate_fn)

    Prior = RNN(voc, num_props=len(property_columns))

    print(f'{property_range} for {property_columns}')
    

    optimizer = torch.optim.Adam(Prior.rnn.parameters(), lr=learning_rate)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=2, threshold=0.5)

    stop_training = False
    loss_log_list = []


    for epoch in range(0, 50):
        if stop_training:
            break

        Prior.rnn.train()
        train_loss = 0
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):

            sequences, properties = batch
            seqs = sequences.long()

            loss = Prior.likelihood(seqs, properties)
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if step % 100 == 0 and step != 0:
                print(f"Epoch {epoch} Step {step} Loss: {loss.item():.2f}")

            sampling_batch_size = 64
            if step == 0 and epoch >= 1:
                tqdm.write("*" * 50)
                ##Adjust ranges based on properties you want to include
                os.makedirs('epoch_model_ckpts', exist_ok=True)
                torch.save(Prior.rnn.state_dict(), f'{updated_model}_epoch{epoch}_step{step}.ckpt')
                print(f"Model saved to: {updated_model}_epoch{epoch}_step{step}.ckpt")
                properties1, properties2 = generate_properties(sampling_batch_size, property_ranges=property_range, scaler=scaler, use_random=True, property_columns=property_columns)
                seqs, _ = Prior.sample(batch_size=sampling_batch_size, properties=properties2, temperature=1) #need number of property vectors to match batch size
                valid = 0
                os.makedirs('epoch_samples_PT_val', exist_ok=True)
                f = open(f'epoch_samples_PT_val/PT_epoch{epoch}_{property_columns}_seed{properties1}.smi', 'a')
                for i, seq in enumerate(seqs.cpu().numpy()):
                    smile = voc.decode(seq)
                    if Chem.MolFromSmiles(smile):
                        valid += 1
                        f.write(smile + "\n")
                    if i < 10:
                        tqdm.write(smile)
                f.close()
                tqdm.write("\n{:>4.1f}% valid SMILES".format(100 * valid / len(seqs)))
                tqdm.write("*" * 50 + "\n")  

            if step == len(train_loader) - 1:
                val_loss = validate(Prior, val_loader)
                #scheduler.step(val_loss)
                print(f"Epoch {epoch} Step {step} Validation Loss: {val_loss:.2f}")


                loss_log_list.append({
                    "Epoch": epoch,
                    "Step": step,
                    "Validation Loss": val_loss,
                    "Training Loss": train_loss / (step + 1)  # Average training loss
                })

                # Early stopping
                early_stopping(val_loss, Prior.rnn, f'{updated_model}_optimal', epoch * len(train_loader) + step)
                if early_stopping.stop:
                    torch.save(Prior.rnn.state_dict(), f'Final_model_epoch{epoch}_step{step}_optimal.ckpt')
                    print("Early stopping")
                    stop_training = True
                    break

        # Save the DataFrame to a CSV file at the end of each epoch
        loss_log_df = pd.DataFrame(loss_log_list)
        loss_log_df.to_csv(f"{updated_model}_loss_log.csv", index=False)

        if stop_training:
            break

    # Ran all epochs without early stopping: save the final model too
    if not stop_training:
        torch.save(Prior.rnn.state_dict(), f'Final_model_epoch{epoch}_step{step}_optimal.ckpt')
        print(f"Model saved to: Final_model_epoch{epoch}_step{step}_optimal.ckpt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train cCLM with specified property columns and filenames.")
    parser.add_argument('--smiles_file', type=str, required=True, help='Path to the input SMILES CSV file.')
    parser.add_argument('--output_csv_scaled', type=str, required=True, help='Path to the output scaled CSV file.')
    parser.add_argument('--output_csv_scaled_cleaned', type=str, required=True, help='Path to the output scaled and cleaned CSV file.')
    parser.add_argument('--scaler_file', type=str, required=True, help='Name of the .pkl file to save the scaler.')
    parser.add_argument('--model_file', type=str, required=True, help='Path to the model file to save the updated model.')
    parser.add_argument('--voc_file', type=str, default="Voc", help='Path to the vocabulary file.')
    parser.add_argument('--property_columns', type=str, nargs='+', default=['binary_activity'],
                        help='List of property columns to use for training.')
    parser.add_argument('--property_range', type=float, nargs='+', metavar=('LOW', 'HIGH'),
                        help='Property values to seed test generations between epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for the optimizer.')

    args = parser.parse_args()

    if args.property_range:
        floats = args.property_range
        if len(floats) % 2 != 0:
            raise ValueError("property_range must contain an even number of values (LOW HIGH per property).")

        # Pair them into (low, high) tuples
        property_range = [(floats[i], floats[i+1]) for i in range(0, len(floats), 2)]

    
        if len(property_range) != len(args.property_columns):
            raise ValueError(
                f"Number of property ranges ({len(property_range)}) must match number of property columns ({len(args.property_columns)})."
            )
    else:
        # default: one property with (0.5, 1.0)
        property_range = [(0.5, 1.0)] * len(args.property_columns)
        
    
    print("Reading smiles...")
    scaled_csv, scaler = ds.scale_and_plot_properties(input_csv=args.smiles_file, output_csv=args.output_csv_scaled, scaler_filename=args.scaler_file, property_columns=args.property_columns)
    smiles_list, updated_df = ds.canonicalize_smiles_from_file(scaled_csv, output_csv=args.output_csv_scaled_cleaned)
    voc_chars = ds.construct_vocabulary(smiles_list, args.voc_file)
    pretrain(input_csv=args.output_csv_scaled_cleaned, property_range=property_range, scaler=scaler, property_columns=args.property_columns, voc_f=args.voc_file, updated_model=args.model_file, batch_size=args.batch_size, learning_rate=args.learning_rate)

