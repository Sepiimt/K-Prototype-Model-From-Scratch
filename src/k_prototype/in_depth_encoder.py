# --- Imports ---
import pandas as pd
import numpy as np

# --- Encoder ---
def in_depth_encoder(data_frame, column_name, separator=","):
    column_series = data_frame[column_name].fillna("Unknown").astype(str)
    # --- Normalizing Tokens ---
    normalized_column_values = column_series.apply(
        lambda x: ", ".join(sorted([token.strip() for token in x.split(separator)]))
    )
    # --- Converting To Category Type ---
    categorical_series = normalized_column_values.astype("category")
    # --- Extracting Codes and Saving Decoder Reference Array ---
    encoded_codes = categorical_series.cat.codes.to_numpy()
    decoder_mapping = categorical_series.cat.categories.to_numpy()
    # --- Return ---
    return encoded_codes, decoder_mapping

# --- Decoder ---
def in_depth_decoder(encoded_codes, decoder_mapping):
    # --- Reverse Integer Codes Back To Original Strings ---
    decoded_output = decoder_mapping[encoded_codes]
    # --- Return ---
    return decoded_output