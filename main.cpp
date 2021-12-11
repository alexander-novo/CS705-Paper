#include <seal/seal.h>

#include <Eigen/Core>
#include <algorithm>
#include <vector>

#include "common.h"

using seal::CoeffModulus, seal::EncryptionParameters, seal::scheme_type, seal::SEALContext, seal::KeyGenerator,
    seal::PublicKey, seal::RelinKeys, seal::GaloisKeys, seal::Encryptor, seal::Evaluator, seal::Decryptor,
    seal::CKKSEncoder, seal::Plaintext, seal::Ciphertext;

// const unsigned N                 = 5;
const size_t poly_modulus_degree = 16384;
const size_t slots               = poly_modulus_degree / 2;
// typedef Eigen::Matrix<double, N, N> Eigen::MatrixXd;
// typedef Eigen::Vector<double, N> Eigen::VectorXd;

void encodeMatrixDiagonal(CKKSEncoder& encoder, double scale, std::vector<Plaintext>& out_mat,
                          const Eigen::MatrixXd& in_mat);

void encodeVector(CKKSEncoder& encoder, double scale, Plaintext& out_vec, const Eigen::VectorXd& in_vec);

void matrixVectorMultiplyDiag(const std::vector<Ciphertext>& mat, unsigned mat_rows, const Ciphertext& vec_in,
                              Ciphertext& vec_out, Evaluator& evaluator, GaloisKeys& gal_keys, RelinKeys& relin_keys);

int main() {
	// Eigen::MatrixXd A = Eigen::MatrixXd::Random();
	// Eigen::VectorXd B = Eigen::VectorXd::Random();

	// std::cout << A << '\n' << B << "\n\n";

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

	std::vector<Eigen::VectorXd> samples;
	std::vector<unsigned> labels;
	std::vector<std::string> trueLabels;

	std::ifstream dataFile("data/iris.data");
	loadData(dataFile, samples, labels, trueLabels);

	std::vector<Eigen::MatrixXd> weights;
	std::vector<Plaintext> weights_plain;
	Plaintext sample_plain;

	std::vector<std::vector<Ciphertext>> weights_enc;
	Ciphertext sample_enc;

	std::ifstream paramFile("out/square_params.dat");
	unsigned count;
	paramFile >> count;
	weights.resize(count);
	weights_enc.resize(count);
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

	std::vector<double> sample;
	for (unsigned i = 0; i < samples.size(); i++) {
		encodeVector(encoder, scale, sample_plain, samples[i]);
		encryptor.encrypt(sample_plain, sample_enc);

		for (unsigned j = 0; j < weights_enc.size(); j++) {
			matrixVectorMultiplyDiag(weights_enc[j], weights[j].rows(), sample_enc, sample_enc, evaluator, gal_keys,
			                         relin_keys);

			evaluator.square_inplace(sample_enc);
			evaluator.relinearize_inplace(sample_enc, relin_keys);
			evaluator.rescale_to_next_inplace(sample_enc);
		}

		decryptor.decrypt(sample_enc, sample_plain);
		encoder.decode(sample_plain, sample);

		for (unsigned i = 0; i < 3; i++) { std::cout << sample[i] << " "; }
		std::cout << labels[i] << std::endl;
	}

	return 0;
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
			diag.resize(in_mat.rows() - i);

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

void encodeVector(CKKSEncoder& encoder, double scale, Plaintext& out_vec, const Eigen::VectorXd& in_vec) {
	std::vector<double> vec(in_vec.rows());
	for (unsigned i = 0; i < in_vec.rows(); i++) { vec[i] = in_vec(i, 0); }

	encoder.encode(vec, scale, out_vec);
}

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