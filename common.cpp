#include "common.h"

template <typename Out>
void split(const std::string& s, char delim, Out result) {
	std::istringstream iss(s);
	std::string item;
	while (std::getline(iss, item, delim)) { *result++ = item; }
}

std::vector<std::string> split(const std::string& s, char delim) {
	std::vector<std::string> elems;
	split(s, delim, std::back_inserter(elems));
	return elems;
}

void loadData(std::ifstream& dataFile, std::vector<Eigen::VectorXd>& samples, std::vector<unsigned>& labels,
              std::vector<std::string>& trueLabels) {
	std::string line;
	while (std::getline(dataFile, line)) {
		if (line.empty()) break;
		std::vector<std::string> pieces = split(line, ',');

		std::string& label = pieces.back();

		std::vector<std::string>::iterator labelIndex = std::find(trueLabels.begin(), trueLabels.end(), label);

		labels.push_back(labelIndex - trueLabels.begin());
		if (labelIndex == trueLabels.end()) { trueLabels.push_back(label); }

		samples.emplace_back(pieces.size() - 1);

		for (unsigned i = 0; i < pieces.size() - 1; i++) { samples.back()(i, 0) = strtod(pieces[i].c_str(), nullptr); }
	}
}
