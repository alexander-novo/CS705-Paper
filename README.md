# CS705-Paper

[Microsoft SEAL](https://github.com/microsoft/SEAL) must be installed.

Running `make` will build both `nn-train` and `main`. `nn-train` will train two neural networks on the iris dataset and dump the parameters into `out/` (it must exist first). Then, `main` will evaluate the network on the dataset while encrypted. 
