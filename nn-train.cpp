#include <Eigen/Core>
#include <Eigen/Geometry>  // For homogenous coordinates
#include <iostream>

#include "common.h"

void nn_train(const std::vector<Eigen::VectorXd>& samples, const std::vector<unsigned>& labels,
              const std::vector<std::string>& trueLabels, const std::vector<unsigned>& shuffle,
              std::vector<Eigen::MatrixXd>& weights,
              const std::vector<Eigen::VectorXd (*)(const Eigen::VectorXd&)>& activations,
              const std::vector<Eigen::MatrixXd (*)(const Eigen::VectorXd&)>& derivatives, unsigned batchSize,
              double learningRate, unsigned total_epochs, std::vector<double>& accuracy,
              std::vector<Eigen::MatrixXd>& bestWeights);
template <typename Out>
void split(const std::string& s, char delim, Out result);
std::vector<std::string> split(const std::string& s, char delim);
Eigen::VectorXd square(const Eigen::VectorXd& x);
Eigen::MatrixXd d_square(const Eigen::VectorXd& x);
Eigen::VectorXd reLU(const Eigen::VectorXd& x);
Eigen::MatrixXd d_reLU(const Eigen::VectorXd& x);
Eigen::VectorXd softmax(const Eigen::VectorXd& x);
Eigen::MatrixXd d_softmax(const Eigen::VectorXd& x);
Eigen::MatrixXd d_loss(const Eigen::VectorXd& y, unsigned label);

int main() {
	std::vector<Eigen::VectorXd> samples;
	std::vector<unsigned> labels;
	std::vector<std::string> trueLabels;
	std::vector<Eigen::VectorXd (*)(const Eigen::VectorXd&)> relu_act, square_act;
	std::vector<Eigen::MatrixXd (*)(const Eigen::VectorXd&)> relu_d, square_d;
	std::vector<double> relu_acc, square_acc;

	std::ifstream dataFile("data/iris.data");
	loadData(dataFile, samples, labels, trueLabels);

	std::ofstream reluParamsFile("out/relu_params.dat");
	std::ofstream squareParamsFile("out/square_params.dat");

	std::vector<unsigned> shuffle(samples.size());
	for (unsigned i = 0; i < shuffle.size(); i++) shuffle[i] = i;

	std::random_shuffle(shuffle.begin(), shuffle.end());

	// Start with two fully-connected layers
	std::vector<Eigen::MatrixXd> init_weights;
	for (unsigned i = 0; i < 2; i++) {
		init_weights.push_back(Eigen::MatrixXd::Random(samples.front().rows(), samples.front().rows()) / 2);
		relu_act.push_back(&reLU);
		relu_d.push_back(&d_reLU);
		square_act.push_back(&square);
		square_d.push_back(&d_square);
	}

	// Then the output layer
	init_weights.push_back(Eigen::MatrixXd::Random(trueLabels.size(), samples.front().rows()) / 2);
	relu_act.push_back(&softmax);
	square_act.push_back(&softmax);
	relu_d.push_back(&d_softmax);
	square_d.push_back(&d_softmax);

	std::vector<Eigen::MatrixXd> weights = init_weights;
	std::vector<Eigen::MatrixXd> bestWeights;

	nn_train(samples, labels, trueLabels, shuffle, weights, relu_act, relu_d, 10, .05, 40, relu_acc, bestWeights);

	reluParamsFile << bestWeights.size() << '\n';
	for (unsigned i = 0; i < bestWeights.size(); i++) {
		reluParamsFile << bestWeights[i].rows() << " " << bestWeights[i].cols() << '\n' << bestWeights[i] << '\n';
	}

	weights = init_weights;

	nn_train(samples, labels, trueLabels, shuffle, weights, square_act, square_d, 5, .005, 40, square_acc, bestWeights);

	squareParamsFile << bestWeights.size() << '\n';
	for (unsigned i = 0; i < bestWeights.size(); i++) {
		squareParamsFile << bestWeights[i].rows() << " " << bestWeights[i].cols() << '\n' << bestWeights[i] << '\n';
	}

	std::cout << "# Accuracy of the classifier per epoch\n# ReLU    Square\n";

	for (unsigned i = 0; i < relu_acc.size(); i++) { std::cout << relu_acc[i] << " " << square_acc[i] << '\n'; }
}

void nn_train(const std::vector<Eigen::VectorXd>& samples, const std::vector<unsigned>& labels,
              const std::vector<std::string>& trueLabels, const std::vector<unsigned>& shuffle,
              std::vector<Eigen::MatrixXd>& weights,
              const std::vector<Eigen::VectorXd (*)(const Eigen::VectorXd&)>& activations,
              const std::vector<Eigen::MatrixXd (*)(const Eigen::VectorXd&)>& derivatives, unsigned batchSize,
              double learningRate, unsigned total_epochs, std::vector<double>& accuracy,
              std::vector<Eigen::MatrixXd>& bestWeights) {
	std::vector<Eigen::VectorXd> trainingX;
	std::vector<Eigen::MatrixXd> trainingY;
	double bestAccuracy = 0;

	for (unsigned epoch = 0; epoch < total_epochs; epoch++) {
		unsigned numCorrect = 0;
		for (unsigned batch = 0; batch < samples.size() / batchSize; batch++) {
			std::vector<Eigen::MatrixXd> tempWeights = weights;
			for (unsigned sample = 0; sample < batchSize; sample++) {
				// for (unsigned sample = 0; sample < 1; sample++) {
				unsigned sample_adj      = shuffle[batch * batchSize + sample];
				Eigen::VectorXd training = samples[sample_adj];

				for (unsigned i = 0; i < weights.size(); i++) {
					trainingX.push_back(training);
					// training = weights[i] * training.homogeneous();
					training = weights[i] * training;
					trainingY.push_back(derivatives[i](training));
					training = activations[i](training);
				}

				bool correct = true;
				for (unsigned i = 0; i < training.rows(); i++) {
					correct &= training(labels[sample_adj], 0) >= training(i, 0);
				}
				if (correct) { numCorrect++; }

				if (epoch > 0) {
					Eigen::MatrixXd D = d_loss(training, labels[sample_adj]);

					for (unsigned i = 0; i < trainingX.size(); i++) {
						unsigned i_rev = trainingX.size() - 1 - i;
						D              = D * trainingY[i_rev];
						// tempWeights[i_rev] -=
						//     (learningRate / batchSize * trainingX[i_rev].homogeneous() * D).transpose();
						tempWeights[i_rev] -= (learningRate / batchSize * trainingX[i_rev] * D).transpose();
						// D = D * weights[i_rev].leftCols(weights[i_rev].cols() - 1);
						D = D * weights[i_rev];
					}
				}

				trainingX.clear();
				trainingY.clear();
			}

			weights = tempWeights;
		}

		accuracy.push_back(numCorrect / (double) samples.size());

		if (accuracy.back() > bestAccuracy) {
			bestAccuracy = accuracy.back();

			bestWeights = weights;
		}
	}
}

Eigen::VectorXd square(const Eigen::VectorXd& x) {
	return x.array().pow(2);
}

Eigen::MatrixXd d_square(const Eigen::VectorXd& x) {
	return (2 * x).asDiagonal();
}

Eigen::VectorXd reLU(const Eigen::VectorXd& x) {
	return x.array().max(0);
}

Eigen::MatrixXd d_reLU(const Eigen::VectorXd& x) {
	return x.unaryExpr([](const double& x) { return x > 0 ? 1.0 : 0.0; }).asDiagonal();
}

Eigen::VectorXd softmax(const Eigen::VectorXd& x) {
	Eigen::VectorXd re(x.rows());

	double sum = 0;

	double max = x.maxCoeff();

	for (unsigned i = 0; i < x.rows(); i++) {
		re(i, 0) = std::exp(x(i, 0) - max);
		sum += re(i, 0);
	}

	re /= sum;

	return re;
}

Eigen::MatrixXd d_softmax(const Eigen::VectorXd& x) {
	Eigen::VectorXd sigma = softmax(x);

	return sigma * -sigma.transpose() + (Eigen::MatrixXd) sigma.asDiagonal();
}

Eigen::MatrixXd d_loss(const Eigen::VectorXd& y, unsigned label) {
	Eigen::MatrixXd re = Eigen::MatrixXd::Zero(1, y.rows());

	re(0, label) = -1 / y(label, 0);

	return re;
}