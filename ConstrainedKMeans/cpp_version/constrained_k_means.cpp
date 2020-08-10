#include <climits>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
#include <fstream>

#include "point.h"
#include "utils.h"

using namespace std;

#define EPS 1e-9

class ConstrainedKMeans {
   private:
    int nClusters, nSamples, nFeatures;
    int* labels;
    int* clusterSizes;
    double* dists;
    double* all_dists;
    std::vector<Point> points;
    std::vector<Point> centroids;
    std::vector<bool> canChange;
    std::string metric;

    void initialize(const vector<double>&, int, int, const vector<bool>&,
                    const vector<int>&);
    void initialize_centroids();
    void e_step();
    void m_step();
    double compute_mse();
    int p3[3];

   public:
    ConstrainedKMeans() {}
    ConstrainedKMeans(int);
    ConstrainedKMeans(int, std::string);

    void fit(const vector<double>&, int, int, const vector<bool>&,
             const vector<int>&);
    int* fit_transform(const vector<double>&, int, int, const vector<bool>&,
                       const vector<int>&);
};

ConstrainedKMeans::ConstrainedKMeans(int nClu) { nClusters = nClu; }

ConstrainedKMeans::ConstrainedKMeans(int nClu, std::string metr) {
    nClusters = nClu;
    metric = metr;
}

void ConstrainedKMeans::initialize_centroids() {
    centroids.reserve(nClusters);

    std::vector<int> indices(nSamples);
    std::iota(indices.begin(), indices.end(), 0);

    // select nClusters random numbers from 0 to nClusters - 1
    std::vector<int> pickedIndices(nClusters);
    std::sample(indices.begin(), indices.end(), pickedIndices.begin(),
                nClusters, std::mt19937{std::random_device{}()});
    for (int i = 0; i < pickedIndices.size(); ++i) {
        centroids[i] = Point(0, nFeatures);
        for (int j = 0; j < nFeatures; ++j)
            centroids[i].coordinates[j] =
                points[pickedIndices[i]].coordinates[j];
    }
}

void ConstrainedKMeans::initialize(const vector<double>& x, int nSamp,
                                   int nFeat, const vector<bool>& mask,
                                   const vector<int>& init_labels) {
    nSamples = nSamp;
    nFeatures = nFeat;
    canChange.reserve(mask.size());
    for (int i = 0; i < nSamp; ++i) canChange[i] = mask[i];

    points.reserve(nSamples);
    for (int i = 0; i < nSamples; ++i) {
        // points[i] = std::move(Point(x + i * nFeatures, nFeatures));
        points[i] = std::move(Point(0, nFeatures));
        for (int j = i * nFeatures, k = 0; j < (i + 1) * nFeatures; ++j, ++k) {
            points[i].coordinates[k] = x[j];
        }
    }

    clusterSizes = new int[nClusters];
    for (int i = 0; i < nClusters; ++i) clusterSizes[i] = 0;

    labels = new int[nSamples];
    for (int i = 0; i < nSamples; ++i) labels[i] = init_labels[i];

    dists = new double[nSamples];
    for (int i = 0; i < nSamples; ++i) dists[i] = 0;

    all_dists = new double[nClusters];
    for (int i = 0; i < nClusters; ++i) all_dists[i] = 0;

    initialize_centroids();
}

void ConstrainedKMeans::e_step() {
    for (int i = 0; i < nSamples; ++i) {
        if (canChange[i]) {
            cdist(points[i], centroids, nClusters, all_dists, metric);
            std::pair<int, double> p = argmin(all_dists, nClusters);
            labels[i] = p.first;
            dists[i] = p.second;
        } else {
            dists[i] = dist_f(metric)(points[i], centroids[labels[i]]);
        }
    }
}

void ConstrainedKMeans::m_step() {
    for (int i = 0; i < nClusters; ++i) centroids[i].fill(0);
    for (int i = 0; i < nClusters; ++i) clusterSizes[i] = 0;

    for (int i = 0; i < nSamples; ++i) {
        centroids[labels[i]] += points[i];
        ++clusterSizes[labels[i]];
    }
    for (int i = 0; i < nClusters; ++i) {
        assert(clusterSizes[i] != 0);
        centroids[i] /= clusterSizes[i];
    }
}

double ConstrainedKMeans::compute_mse() {
    double mse = 0;
    for (int i = 0; i < nSamples; ++i) mse += dists[i] * dists[i];
    mse /= nSamples;
    return mse;
}

void ConstrainedKMeans::fit(const vector<double>& x, int nSamp, int nFeat,
                            const vector<bool>& mask,
                            const vector<int>& init_labels) {
    /*
     * If mask is 0, don't change label.
     */
    initialize(x, nSamp, nFeat, mask, init_labels);

    double mse = -1, new_mse;
    int steps = 0;

    while (true) {
        ++steps;
        e_step();
        m_step();
        // printf("Centroids are\n");
        // for (int i = 0; i < nClusters; ++i) cout << centroids[i];
        new_mse = compute_mse();
        printf("Iteration %d :: MSE %f\n", steps, new_mse);
        if (std::abs(new_mse - mse) < EPS) break;
        mse = new_mse;
    }

    printf("Converged in %i steps.\n", steps);
}

int* ConstrainedKMeans::fit_transform(const vector<double>& x, int nSamp,
                                      int nFeat, const vector<bool>& mask,
                                      const vector<int>& init_labels) {
    fit(x, nSamp, nFeat, mask, init_labels);
    return labels;
}

int main() {
    ConstrainedKMeans ckm(6, "euclidean");
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(0, 20);

    const int n = 8000, m = 2;
    vector<bool> mask(n);
    vector<int> init_labels(n);
    vector<double> x(n * m);
    cout << "Generating random data...\n";
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) x[i * m + j] = double(dist(e2));
    for (int i = 0; i < n; ++i) mask[i] = 1;
    for (int i = 0; i < n; ++i) init_labels[i] = int(dist(e2)) % 4;
    cout << "Fitting...\n";
    int* labels = ckm.fit_transform(x, n, m, mask, init_labels);

    cout << "Writing to file...\n";
    ofstream myfile;
    myfile.open ("labels.txt");
    for (int i = 0; i < n; ++i) myfile << labels[i] << ",";
    myfile.close();
    ofstream x_file;
    x_file.open ("x.txt");
    for (int i = 0; i < n*m; ++i) x_file << x[i] << ",";
    x_file.close();
    return 0;
}

//#include <boost/foreach.hpp>
//#include <boost/python.hpp>
//#include <boost/python/def.hpp>
//#include <boost/python/extract.hpp>
//#include <boost/python/list.hpp>
//#include <boost/python/module.hpp>
//#include <boost/python/numpy.hpp>
//#include <boost/python/stl_iterator.hpp>
//
// namespace p = boost::python;
// namespace np = boost::python::numpy;
//
// np::ndarray set_x(const boost::python::list& pylist, int w, int h) {
//    int counter = 0;
//    vector<double> x(w * h);
//    typedef boost::python::stl_input_iterator<double> iterator_type;
//    BOOST_FOREACH (const iterator_type::value_type& entry,
//                   std::make_pair(iterator_type(pylist), iterator_type())) {
//        x[counter++] = double(entry);
//    }
//    ConstrainedKMeans ckm(3, "euclidean");
//    int* labels = new int[w];
//    labels = ckm.fit_transform(x, w, h);
//    for (int i = 0; i < w; ++i) cout << labels[i] << " ";
//    cout << endl;
//}
//
// BOOST_PYTHON_MODULE(ckm) {
//    namespace python = boost::python;
//    python::def("set_x", &set_x);
//}
