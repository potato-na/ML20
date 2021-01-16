import h5py
import numpy as np

def create_hdf5():
	with h5py.File("mytestfile.hdf5", "w") as f:
		dset = f.create_dataset("mydataset", (100, ), dtype="i")

def read_hdf5():
	f = h5py.File("mytestfile.hdf5", "r")

	# like Python dictionary
	print(list(f.keys()))
	dset = f['mydataset']

	# like numpy arrays
	print(dset.shape)
	print(dset.dtype)

	# array-style slicing
	dset = np.arange(100)
	print(dset[0])
	print(dset[10])
	print(dset[0:100:10])

	return f

def group_hdf5():
	f = read_hdf5()
	dset = f['mydataset']

	# Every object has a name
	# POSIX-style hierarchy with `/` separators
	print(dset.name)

	# "folders" are called groups
	# `File` object is itself a group, in this case the root group, named `/`
	print(f.name)

def printname(name):
	print(name)

def create_subgroup():
	create_hdf5()
	# open the file in the "append" mode
	f = h5py.File('mytestfile.hdf5', 'a')
	grp = f.create_group("subgroup")

	print(f.keys())

	# all `Group` Objects also have `create_*` methods

	dset2 = grp.create_dataset("another_dataset", (50,), dtype="f")
	print(dset2.name)

	dset3 = f.create_dataset("subgroup2/dataset_three", (10,), dtype="i")
	print(dset3.name)

	# Groups support most of the Python dictionary-style interface
	dataset_three = f["subgroup2/dataset_three"]

	# Iterating
	print()
	print('names of its members')
	for name in f:
		print(name)

	# iterating over an entire file
	print()
	print('iterating over an entire file')
	f.visit(printname)

	# Membership testing
	print()
	print('is my dataset in f:', 'mydataset' in f)
	print('is somethingelese in f:', 'somethingelese' in f)
	print('is subgroup/another_dataset in f', 'subgroup/another_dataset:' in f)

def attributes():
	create_hdf5()
	f = read_hdf5()
	dset = f['mydataset']
	dset.attrs['temperature'] = 99.5
	print(dset.attrs['temperature'])
	print('temperature in dset.attrs: ', 'temperature' in dset.attrs)



def main():
	# create_hdf5()
	# read_hdf5()
	# group_hdf5()
	create_subgroup()
	# attributes()


if __name__ == "__main__":
	main()