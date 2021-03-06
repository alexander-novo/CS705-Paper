#include <seal/seal.h>

#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <vector>

#include "common.h"

using seal::CoeffModulus, seal::EncryptionParameters, seal::scheme_type, seal::SEALContext, seal::KeyGenerator,
    seal::PublicKey, seal::RelinKeys, seal::GaloisKeys, seal::Encryptor, seal::Evaluator, seal::Decryptor,
    seal::CKKSEncoder, seal::Plaintext, seal::Ciphertext;

const size_t poly_modulus_degree = 16384;
const size_t slots               = poly_modulus_degree / 2;

#pragma region Function Declarations
void compareMatrixEncodings(CKKSEncoder& encoder, Evaluator& evaluator, Encryptor& encryptor, Decryptor& decryptor,
                            GaloisKeys& gal_keys, RelinKeys& relin_keys, double scale, unsigned max_d);

void encodeMatrixRows(CKKSEncoder& encoder, double scale, std::vector<Plaintext>& out_mat,
                      const Eigen::MatrixXd& in_mat);

void encodeMatrixDiagonal(CKKSEncoder& encoder, double scale, std::vector<Plaintext>& out_mat,
                          const Eigen::MatrixXd& in_mat);

void encodeMatrixDiagonalSIMD(CKKSEncoder& encoder, double scale, std::vector<Plaintext>& out_mat,
                              const Eigen::MatrixXd& in_mat, unsigned ammountSIMD, unsigned maxWidth);

void encodeVector(CKKSEncoder& encoder, double scale, Plaintext& out_vec, const Eigen::VectorXd& in_vec);

void encodeVectorSIMD(CKKSEncoder& encoder, double scale, Plaintext& out_vec,
                      const std::vector<Eigen::VectorXd>& in_vec, unsigned maxWidth);

void matrixVectorMultiplyRows(const std::vector<Ciphertext>& mat, unsigned mat_cols, const Ciphertext& vec_in,
                              Ciphertext& vec_out, Evaluator& evaluator, Plaintext unitMult, GaloisKeys& gal_keys,
                              RelinKeys& relin_keys);

void matrixVectorMultiplyDiag(const std::vector<Ciphertext>& mat, unsigned mat_rows, const Ciphertext& vec_in,
                              Ciphertext& vec_out, Evaluator& evaluator, GaloisKeys& gal_keys, RelinKeys& relin_keys);

#pragma endregion

int main() {
	std::cout << "Begin Encryption Setup & Data Loading..." << std::endl;
	auto start = std::chrono::high_resolution_clock::now();

#pragma region Encryption parameters
	EncryptionParameters params(scheme_type::ckks);

	params.set_poly_modulus_degree(poly_modulus_degree);

	// Max coeff_modulus for poly_modulus = 16384 is 438
	// We want the end primes to be around 60, so we have a budget of 318 left, split amongst 10*30
	params.set_coeff_modulus(
	    CoeffModulus::Create(poly_modulus_degree, {60, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 60}));

	// Similar to the middle prime bits above
	double scale = pow(2.0, 30);

	SEALContext context(params);
	KeyGenerator keygen(context);
	auto secret_key = keygen.secret_key();
	PublicKey public_key;
	keygen.create_public_key(public_key);
	RelinKeys relin_keys;
	keygen.create_relin_keys(relin_keys);
	GaloisKeys gal_keys;
	keygen.create_galois_keys(gal_keys);
	Encryptor encryptor(context, public_key);
	Evaluator evaluator(context);
	Decryptor decryptor(context, secret_key);
	CKKSEncoder encoder(context);
#pragma endregion

#pragma region Loading, encoding, and encrypting data
	std::vector<Eigen::VectorXd> samples;
	std::vector<unsigned> labels;
	std::vector<std::string> trueLabels;

	std::ifstream dataFile("data/iris.data");
	loadData(dataFile, samples, labels, trueLabels);

	std::vector<Eigen::MatrixXd> weights;
	std::vector<Plaintext> weights_plain;
	std::vector<Plaintext> weights_plainSIMD;
	Plaintext sample_plain;
	Plaintext sample_plainSIMD;

	std::vector<std::vector<Ciphertext>> weights_enc;
	std::vector<std::vector<Ciphertext>> weights_encSIMD;
	std::vector<Ciphertext> sample_enc;
	Ciphertext sample_encSIMD;

	std::ifstream paramFile("out/square_params.dat");
	unsigned count;
	paramFile >> count;
	weights.resize(count);
	weights_enc.resize(count);
	weights_encSIMD.resize(count);
	for (unsigned i = 0; i < count; i++) {
		unsigned rows, cols;
		paramFile >> rows >> cols;

		weights[i].resize(rows, cols);

		for (unsigned row = 0; row < rows; row++) {
			for (unsigned col = 0; col < cols; col++) {
				double val;
				paramFile >> val;

				weights[i](row, col) = val;
			}
		}

		encodeMatrixDiagonal(encoder, scale, weights_plain, weights[i]);

		weights_enc[i].resize(weights_plain.size());
		for (unsigned j = 0; j < weights_plain.size(); j++) { encryptor.encrypt(weights_plain[j], weights_enc[i][j]); }
	}

	unsigned maxWidth = samples.front().rows();

	for (unsigned i = 0; i < weights.size(); i++) { maxWidth = std::max(maxWidth, (unsigned) weights[i].rows()); }
	for (unsigned i = 0; i < weights.size(); i++) {
		encodeMatrixDiagonalSIMD(encoder, scale, weights_plainSIMD, weights[i], samples.size(), maxWidth);

		weights_encSIMD[i].resize(weights_plainSIMD.size());
		for (unsigned j = 0; j < weights_plainSIMD.size(); j++) {
			encryptor.encrypt(weights_plainSIMD[j], weights_encSIMD[i][j]);
		}
	}

	encodeVectorSIMD(encoder, scale, sample_plainSIMD, samples, maxWidth);
	encryptor.encrypt(sample_plainSIMD, sample_encSIMD);

	sample_enc.resize(samples.size());
	for (unsigned i = 0; i < samples.size(); i++) {
		encodeVector(encoder, scale, sample_plain, samples[i]);
		encryptor.encrypt(sample_plain, sample_enc[i]);
	}
#pragma endregion

	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() -
	                                                                   start)
	                     .count() /
	                 1e6
	          << "s taken.\nBegin Matrix Encoding Comparison..." << std::endl;
	start = std::chrono::high_resolution_clock::now();

	compareMatrixEncodings(encoder, evaluator, encryptor, decryptor, gal_keys, relin_keys, scale, 100);

	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() -
	                                                                   start)
	                     .count() /
	                 1e6
	          << "s taken.\nBegin Baseline Feedforward Evaluation..." << std::endl;
	start = std::chrono::high_resolution_clock::now();

#pragma region Evaluate Baseline Feedforward
	Eigen::VectorXd feed;
	unsigned numCorrect = 0;
	for (unsigned i = 0; i < samples.size(); i++) {
		feed = samples[i];

		unsigned w;
		for (w = 0; w < weights.size() - 1; w++) { feed = (weights[w] * feed).array().square(); }
		feed = weights[w] * feed;

		unsigned maxIndex = 0;
		for (unsigned j = 1; j < 3; j++) {
			if (feed(j, 0) > feed(maxIndex, 0)) maxIndex = j;
		}
		if (maxIndex == labels[i]) numCorrect++;
	}

	std::cout << "Baseline accuracy: " << numCorrect / (double) samples.size() << std::endl;
#pragma endregion

	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() -
	                                                                   start)
	                     .count() /
	                 1e6
	          << "s taken.\nBegin SIMD Feedforward Evaluation..." << std::endl;
	start = std::chrono::high_resolution_clock::now();

#pragma region Evaluate SIMD Encrypted Feedforward
	std::vector<double> sample;

	unsigned j;
	for (j = 0; j < weights_enc.size() - 1; j++) {
		matrixVectorMultiplyDiag(weights_encSIMD[j], weights[j].rows(), sample_encSIMD, sample_encSIMD, evaluator,
		                         gal_keys, relin_keys);

		evaluator.square_inplace(sample_encSIMD);
		evaluator.relinearize_inplace(sample_encSIMD, relin_keys);
		evaluator.rescale_to_next_inplace(sample_encSIMD);
	}

	// Don't apply square activation to last layer
	matrixVectorMultiplyDiag(weights_encSIMD[j], weights[j].rows(), sample_encSIMD, sample_encSIMD, evaluator, gal_keys,
	                         relin_keys);

	decryptor.decrypt(sample_encSIMD, sample_plainSIMD);
	encoder.decode(sample_plainSIMD, sample);

	numCorrect = 0;
	for (unsigned i = 0; i < samples.size(); i++) {
		bool correct = true;
		for (unsigned j = 0; j < 3; j++) { correct &= sample[i * maxWidth + labels[i]] >= sample[i * maxWidth + j]; }

		if (correct) numCorrect++;
	}

	std::cout << "SIMD Accuracy: " << (numCorrect / (double) samples.size()) << std::endl;
#pragma endregion

	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() -
	                                                                   start)
	                     .count() /
	                 1e6
	          << "s taken.\nBegin Normal Encrypted Feedforward Evaluation..." << std::endl;
	start = std::chrono::high_resolution_clock::now();

#pragma region Evaluate Normal Encrypted Feedforward
	numCorrect = 0;

	for (unsigned i = 0; i < samples.size(); i++) {
		unsigned j;
		for (j = 0; j < weights_enc.size() - 1; j++) {
			matrixVectorMultiplyDiag(weights_enc[j], weights[j].rows(), sample_enc[i], sample_enc[i], evaluator,
			                         gal_keys, relin_keys);

			evaluator.square_inplace(sample_enc[i]);
			evaluator.relinearize_inplace(sample_enc[i], relin_keys);
			evaluator.rescale_to_next_inplace(sample_enc[i]);
		}

		// Don't apply square activation to last layer
		matrixVectorMultiplyDiag(weights_enc[j], weights[j].rows(), sample_enc[i], sample_enc[i], evaluator, gal_keys,
		                         relin_keys);

		decryptor.decrypt(sample_enc[i], sample_plain);
		encoder.decode(sample_plain, sample);

		unsigned maxIndex = 0;
		for (unsigned j = 1; j < 3; j++) {
			if (sample[j] > sample[maxIndex]) maxIndex = j;
		}
		if (maxIndex == labels[i]) numCorrect++;
	}

	std::cout << "Normal Accuracy: " << (numCorrect / (double) samples.size()) << std::endl;
#pragma endregion

	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() -
	                                                                   start)
	                     .count() /
	                 1e6
	          << "s taken." << std::endl;

	return 0;
}

void compareMatrixEncodings(CKKSEncoder& encoder, Evaluator& evaluator, Encryptor& encryptor, Decryptor& decryptor,
                            GaloisKeys& gal_keys, RelinKeys& relin_keys, double scale, unsigned max_d) {
	std::ofstream out("out/matrix-encoding-comp.dat");
	Eigen::MatrixXd A;
	Eigen::VectorXd x, y, y_test;

	std::vector<Plaintext> A_plain;
	Plaintext x_plain, y_plain;

	std::vector<Ciphertext> A_enc;
	Ciphertext x_enc, y_enc;

	// unitMult is the vector <1, 0, 0, 0, ..., 0> for the purposes of performing matrix-vector multiplication with
	// row-encoded matrices.
	std::vector<double> unit = {1.0};
	Plaintext unitMult;

	std::vector<double> test_out;

	encoder.encode(unit, scale, unitMult);

	out << "# Comparison of time (in seconds) it takes to multiply a dxd matrix by a d-dimensional vector based on "
	       "encoding\nd   Row-Time   Row-Error   Diagonal-Time   Diagonal-Error\n";
	for (unsigned d = 1; d <= max_d; d++) {
		A = Eigen::MatrixXd::Random(d, d);
		x = Eigen::VectorXd::Random(d);
		y = A * x;
		y_test.resizeLike(y);

		out << d << ' ';

		// Vector encoding is the same
		encodeVector(encoder, scale, x_plain, x);
		encryptor.encrypt(x_plain, x_enc);

		// Row Encoding
		encodeMatrixRows(encoder, scale, A_plain, A);
		A_enc.resize(A_plain.size());
		for (unsigned i = 0; i < A_plain.size(); i++) encryptor.encrypt(A_plain[i], A_enc[i]);

		auto start = std::chrono::high_resolution_clock::now();
		matrixVectorMultiplyRows(A_enc, d, x_enc, y_enc, evaluator, unitMult, gal_keys, relin_keys);
		out << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start)
		               .count() /
		           1e6
		    << ' ';

		decryptor.decrypt(y_enc, y_plain);
		encoder.decode(y_plain, test_out);

		for (unsigned i = 0; i < d; i++) y_test(i, 0) = test_out[i];
		out << (y - y_test).norm() << ' ';

		// Diagonal encoding
		encodeMatrixDiagonal(encoder, scale, A_plain, A);
		A_enc.resize(A_plain.size());
		for (unsigned i = 0; i < A_plain.size(); i++) encryptor.encrypt(A_plain[i], A_enc[i]);

		start = std::chrono::high_resolution_clock::now();
		matrixVectorMultiplyDiag(A_enc, d, x_enc, y_enc, evaluator, gal_keys, relin_keys);
		out << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start)
		               .count() /
		           1e6
		    << ' ';

		decryptor.decrypt(y_enc, y_plain);
		encoder.decode(y_plain, test_out);

		for (unsigned i = 0; i < d; i++) y_test(i, 0) = test_out[i];
		out << (y - y_test).norm() << '\n';

		if (d % 10 == 0 || max_d - d < 10) std::cout << d << " complete.\n";
	}
}

void encodeMatrixRows(CKKSEncoder& encoder, double scale, std::vector<Plaintext>& out_mat,
                      const Eigen::MatrixXd& in_mat) {
	out_mat.resize(in_mat.rows());
	std::vector<double> row(in_mat.cols());
	for (unsigned i = 0; i < in_mat.rows(); i++) {
		for (unsigned j = 0; j < in_mat.cols(); j++) { row[j] = in_mat(i, j); }

		encoder.encode(row, scale, out_mat[i]);
	}
}

void encodeMatrixDiagonal(CKKSEncoder& encoder, double scale, std::vector<Plaintext>& out_mat,
                          const Eigen::MatrixXd& in_mat) {
	// Only store the non-zero diagonals. It's assumed that all of the diagonals in in_mat are non-zero.
	// For an MxN matrix stored in a larger `slots`x`slots` matrix, there are a maximum of `slots` diagonals (one for
	// each column of the larger matrix), but many of these diagonals may be zero. In that case, each row, column in the
	// smaller input matrix is the beginning of the non-zero part of a diagonal, and there are N + M - 1 (-1 for the
	// main diagonal which is the beginning of a row and a column) nonzero diagonals. There are no fewer, as if a
	// nonzero diagonal starts at a column of the smaller input matrix and continues along a row of the matrix, then
	// there are no zero diagonals (as every subsequent diagonal continues off the rows of the input matrix as well).
	out_mat.resize(std::min(slots, (size_t) (in_mat.rows() + in_mat.cols() - 1)));
	std::vector<double> diag;
	// If there are zero diagonals, we don't want to calculate/store those zero diagonals, so we instead store the last
	// nonzero diagonals (the ones below the main diagonal in the input matrix) before the initial diagonal (the main
	// diagonal)
	if (out_mat.size() < slots) {
		// In this case, all of the diagonals are of max size M. For the diagonals below the main diagonal (and the main
		// diagonal), they are exactly M. Every diagonal after the main diagonal shrinks by 1.
		diag.resize(in_mat.rows());
		// These diagonals are in reverse oder from the main diagonal out. So we start at the bottom row of the
		// matrix and work our way up. i is the starting row of the nonzero part of the diagonal
		for (unsigned i = in_mat.rows() - 1; i > 0; i--) {
			unsigned j;
			// They start with a number of zeros in every row before this one. j is the current row
			for (j = 0; j < i; j++) { diag[j] = 0; }
			// The column of the diagonal is the current row minus the starting row
			for (; j < in_mat.rows(); j++) { diag[j] = in_mat(j, j - i); }

			encoder.encode(diag, scale, out_mat[in_mat.rows() - 1 - i]);
		}

		// Now the normal diagonals. i is the column of the first element in the diagonal
		for (unsigned i = 0; i < in_mat.cols(); i++) {
			// If in_mat is wide, the length of the diagonals will continue to be the number of rows for a while.
			// Otherwise, the diagonal will reduce in size by one
			if (i < in_mat.cols() - in_mat.rows()) diag.resize(in_mat.rows() - i);

			unsigned j;
			// j is the element in the column (also the row of the element) and i + j is the column
			for (j = 0; j < in_mat.rows() && i + j < in_mat.cols(); j++) { diag[j] = in_mat(j, i + j); }

			// In this case, every diagonal is distinct, so we know the rest of the elements are zero.
			encoder.encode(diag, scale, out_mat[i + in_mat.rows() - 1]);
		}
	} else {
		// In this case, we know that every diagonal is nonzero, so we can just calculate as normal.
		for (unsigned i = 0; i < slots; i++) {
			unsigned j;
			for (j = 0; j < in_mat.rows() && i + j < in_mat.cols(); j++) { diag.push_back(in_mat(j, i + j)); }

			// slots - (j + i) is how many more rows we need to travel before looping back around to the first column,
			// so j + slots - (j + i) = slots - i is the first row which may be back in the input matrix.
			// If it *is* in the input matrix, we need to add zeros before adding the rest of the diagonal vector.
			if (slots - i < in_mat.rows()) {
				for (; j < slots - i; j++) { diag.push_back(0); }

				// Finish adding the rest of the non-zero elements in the diagonal vector. We don't necessarily need to
				// finish the whole diagonal vector, since the encoder will do that for us.
				for (unsigned k = 0; j < in_mat.rows(); j++, k++) { diag.push_back(in_mat(j, k)); }
			}

			encoder.encode(diag, scale, out_mat[i]);
			diag.clear();
		}
	}

	unsigned offset = out_mat.size() < slots ? in_mat.rows() - 1 : 0;
}

void encodeMatrixDiagonalSIMD(CKKSEncoder& encoder, double scale, std::vector<Plaintext>& out_mat,
                              const Eigen::MatrixXd& in_mat, unsigned ammountSIMD, unsigned maxWidth) {
	// Still same number of diagonals as non-SIMD, since the matrices are all stored on the diagonal
	out_mat.resize(std::min(slots, (size_t) (in_mat.rows() + in_mat.cols() - 1)));
	std::vector<double> diag;
	// If there are zero diagonals, we don't want to calculate/store those zero diagonals, so we instead store the last
	// nonzero diagonals (the ones below the main diagonal in the input matrix) before the initial diagonal (the main
	// diagonal)

	// Each diagonal below the main diagonal is the same size as before, but replicated some number of times for
	// SIMD. The replication process is what increases the size. To simplify this process, diagonals above the main
	// diagonal will store a couple of extra zeros at the end. Therefore every diagonal is the same size
	diag.resize(maxWidth);

	if (out_mat.size() < slots) {
		for (unsigned i = in_mat.rows() - 1; i > 0; i--) {
			unsigned j;
			for (j = 0; j < i; j++) { diag[j] = 0; }
			for (; j < in_mat.rows(); j++) { diag[j] = in_mat(j, j - i); }
			for (; j < maxWidth; j++) { diag[j] = 0; }

			// Diagonal is encoded as before, but now we replicate it some number of times
			for (unsigned n = 0; n < ammountSIMD; n++) {
				diag.insert(diag.end(), diag.begin(), diag.begin() + maxWidth);
			}

			encoder.encode(diag, scale, out_mat[in_mat.rows() - 1 - i]);
			diag.resize(maxWidth);
		}

		for (unsigned i = 0; i < in_mat.cols(); i++) {
			unsigned j;
			for (j = 0; j < in_mat.rows() && i + j < in_mat.cols(); j++) { diag[j] = in_mat(j, i + j); }

			// Same as before, but we fill out the rest of the zeros for easy replication
			for (; j < maxWidth; j++) { diag[j] = 0; }

			for (unsigned n = 0; n < ammountSIMD; n++) {
				diag.insert(diag.end(), diag.begin(), diag.begin() + maxWidth);
			}

			encoder.encode(diag, scale, out_mat[i + in_mat.rows() - 1]);
			diag.resize(maxWidth);
		}
	} else {
		for (unsigned i = 0; i < slots; i++) {
			unsigned j;
			for (j = 0; j < in_mat.rows() && i + j < in_mat.cols(); j++) { diag[j] = in_mat(j, i + j); }

			if (slots - i < in_mat.rows()) {
				for (; j < slots - i; j++) { diag[j] = 0; }
				for (unsigned k = 0; j < in_mat.rows(); j++, k++) { diag[j] = in_mat(j, k); }
			}

			for (; j < maxWidth; j++) { diag[j] = 0; }

			for (unsigned n = 0; n < ammountSIMD; n++) {
				diag.insert(diag.end(), diag.begin(), diag.begin() + maxWidth);
			}

			encoder.encode(diag, scale, out_mat[i]);
			diag.resize(maxWidth);
		}
	}
}

void encodeVector(CKKSEncoder& encoder, double scale, Plaintext& out_vec, const Eigen::VectorXd& in_vec) {
	std::vector<double> vec(in_vec.rows());
	for (unsigned i = 0; i < in_vec.rows(); i++) { vec[i] = in_vec(i, 0); }

	encoder.encode(vec, scale, out_vec);
}

void encodeVectorSIMD(CKKSEncoder& encoder, double scale, Plaintext& out_vec,
                      const std::vector<Eigen::VectorXd>& in_vec, unsigned maxWidth) {
	std::vector<double> vec(maxWidth * in_vec.size(), 0.0);
	for (unsigned i = 0; i < in_vec.size(); i++) {
		unsigned j;
		for (j = 0; j < in_vec[i].rows(); j++) { vec[i * maxWidth + j] = in_vec[i](j, 0); }
		for (; j < maxWidth; j++) { vec[i * maxWidth + j] = 0; }
	}

	encoder.encode(vec, scale, out_vec);
}

/**
 * @brief Perform encrypted matrix-vector multiplication with a row encoded matrix.
 *
 * @param      mat        The matrix to multiply. Should be row encoded with encodeMatrixRows();
 * @param      mat_cols   Number of columns in mat.
 * @param      vec_in     The vector to multiply.
 * @param[out] vec_out    The result of the multiplication.
 * @param      evaluator  The evaluator used to evaluate the math.
 * @param      unitMult   The vector <1, 0, 0, 0, ..., 0> used for multiplication.
 * @param      gal_keys   Galois keys needed to perform rotations.
 * @param      relin_keys Relineraization keys needed for relineraization after multiplications.
 */
void matrixVectorMultiplyRows(const std::vector<Ciphertext>& mat, unsigned mat_cols, const Ciphertext& vec_in,
                              Ciphertext& vec_out, Evaluator& evaluator, Plaintext unitMult, GaloisKeys& gal_keys,
                              RelinKeys& relin_keys) {
	// Each row-vector times the input vector corresponds to exactly one slot of the output, so our summands will be
	// zero everywhere except their specific slot.
	std::vector<Ciphertext> summands(mat.size());
	Ciphertext temp_rotate;

	for (unsigned i = 0; i < mat.size(); i++) {
		// Take the dot product between a row vector and the input vector.
		// The first step is the element-wise product.
		evaluator.mod_switch_to(mat[i], vec_in.parms_id(), summands[i]);
		evaluator.multiply_inplace(summands[i], vec_in);
		evaluator.relinearize_inplace(summands[i], relin_keys);
		evaluator.rescale_to_next_inplace(summands[i]);

		// Next we must sum the elements. We do thise by rotating the current sum so that different elements overlap and
		// do an elementwise sum. We rotate in a smart way so that we can use previous sums and only do log_2(d)
		// rotations, where d is the dimensionality of the data. However, our vectors are actually very large vectors of
		// mostly zeroes, so these zeroes "infect" the sum vector, leaving only the first slot correct. Therefore we
		// need to multiply this by the vector <1, 0, 0, 0, ...> and rotate it so that the answer is in the correct
		// slot.
		for (unsigned r = 1; r < mat_cols; r *= 2) {
			// By rotating our in-progress sum (rather than the initial dot product vector), we can make use of the
			// previous sums
			evaluator.rotate_vector(summands[i], r, gal_keys, temp_rotate);
			evaluator.add_inplace(summands[i], temp_rotate);
		}

		evaluator.mod_switch_to(unitMult, summands[i].parms_id(), unitMult);
		evaluator.multiply_plain_inplace(summands[i], unitMult);
		evaluator.rotate_vector_inplace(summands[i], -i, gal_keys);
		evaluator.relinearize_inplace(summands[i], relin_keys);
		evaluator.rescale_to_next_inplace(summands[i]);
	}

	// Now our summands look like a collection of vectors:
	// <a, 0, 0, 0, ..., 0>,
	// <0, b, 0, 0, ..., 0>,
	// <0, 0, c, 0, ..., 0>...
	// where a, b, c, ... are the correct dot product of the corresponding row in the matrix and the input vector.
	// We just sum them to end up with <a, b, c, ...>, which is the diefinition of the matrix-vector product.
	evaluator.add_many(summands, vec_out);
}

/**
 * @brief Perform encrypted matrix-vector multiplication with a diagonal encoded matrix.
 *
 * @param      mat        The matrix to multiply. Should be diagonal encoded with encodeMatrixDiagonal() or
 *                        encodeMatrixDiagonalSIMD().
 * @param      mat_rows   Number of rows in mat.
 * @param      vec_in     The vector to multiply.
 * @param[out] vec_out    The result of the multiplication.
 * @param      evaluator  The evaluator used to evaluate the math.
 * @param      gal_keys   Galois keys needed to perform rotations.
 * @param      relin_keys Relineraization keys needed for relineraization after multiplications.
 */
void matrixVectorMultiplyDiag(const std::vector<Ciphertext>& mat, unsigned mat_rows, const Ciphertext& vec_in,
                              Ciphertext& vec_out, Evaluator& evaluator, GaloisKeys& gal_keys, RelinKeys& relin_keys) {
	Ciphertext rotate = vec_in;
	std::vector<Ciphertext> summands(mat.size());

	if (mat.size() < slots) { evaluator.rotate_vector_inplace(rotate, -(mat_rows - 1), gal_keys); }

	unsigned i;
	Ciphertext matDiag;
	for (i = 0; i < mat.size() - 1; i++) {
		evaluator.mod_switch_to(mat[i], rotate.parms_id(), matDiag);
		evaluator.multiply(matDiag, rotate, summands[i]);
		evaluator.rotate_vector_inplace(rotate, 1, gal_keys);
	}

	evaluator.mod_switch_to(mat[i], rotate.parms_id(), matDiag);
	evaluator.multiply(matDiag, rotate, summands[i]);

	evaluator.add_many(summands, vec_out);
	evaluator.relinearize_inplace(vec_out, relin_keys);
	evaluator.rescale_to_next_inplace(vec_out);
}