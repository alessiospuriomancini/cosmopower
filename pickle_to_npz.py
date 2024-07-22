import pickle
import numpy as np
import argparse
import os
import sys

argp = argparse.ArgumentParser(description = "Re-format old cosmopower Pickle files to npz files.")
argp.add_argument("filename", type = str, help = "The path to the .pkl file (without extension).")
argp.add_argument("-p", "--pca-nn", dest = "PCAplusNN", action = "store_true", help = "The network is a PCA+NN type network.")
args = argp.parse_args()

fn, ext = os.path.splitext(args.filename)
if ext != "":
	print("Provide the filename WITHOUT the extension!")
	sys.exit()

with open(fn + ".pkl", "rb") as fp:
	if args.PCAplusNN:
		W, b, alphas, betas, parameters_mean, parameters_std, pca_mean, pca_std, features_mean, features_std, parameters, n_parameters, modes, n_modes, n_pcas, pca_transform_matrix, n_hidden, n_layers, architecture = pickle.load(fp)
	else:
		W, b, alphas, betas, parameters_mean, parameters_std, features_mean, features_std, n_parameters, parameters, n_modes, modes, n_hidden, n_layers, architecture = pickle.load(fp)
	
	attributes = { }
	attributes["architecture"] = architecture
	attributes["n_layers"] = n_layers
	attributes["n_hidden"] = n_hidden
	attributes["n_parameters"] = n_parameters
	attributes["n_modes"] = n_modes

	attributes["parameters"] = parameters
	attributes["modes"] = modes

	attributes["parameters_mean"] = parameters_mean
	attributes["parameters_std"] = parameters_std
	attributes["features_mean"] = features_mean
	attributes["features_std"] = features_std

	for i in range(n_layers):
		attributes[f"W_{i}"] = W[i]
		attributes[f"b_{i}"] = b[i]
	for i in range(n_layers-1):
		attributes[f"alphas_{i}"] = alphas[i]
		attributes[f"betas_{i}"] = betas[i]
	
	if args.PCAplusNN:
		attributes["pca_mean"] = pca_mean
		attributes["pca_std"] = pca_std
		attributes["n_pcas"] = n_pcas
		attributes["pca_transform_matrix"] = pca_transform_matrix

with open(fn + ".npz", "wb") as fp:
	np.savez_compressed(fp, **attributes)
