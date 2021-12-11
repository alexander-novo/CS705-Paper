#include <Eigen/Core>
#include <fstream>
#include <string>
#include <vector>

void loadData(std::ifstream& dataFile, std::vector<Eigen::VectorXd>& samples, std::vector<unsigned>& labels,
              std::vector<std::string>& trueLabels);