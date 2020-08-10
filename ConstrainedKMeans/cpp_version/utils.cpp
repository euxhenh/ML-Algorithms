#include "utils.h"

#include "point.h"

double euclidean(const Point& p1, const Point& p2) {
    Point p = p1 - p2;
    return std::sqrt((p * p).sum());
}

std::function<double(const Point&, const Point&)> dist_f(std::string metric) {
    if (metric == "euclidean") return &euclidean;
    throw std::invalid_argument("Invalid metric" + metric + " encountered.");
}

std::pair<int, double> argmin(double* x, int n) {
    int minIndex = 0;
    double minDist = *x;

    for (int i = 1; i < n; ++i) {
        if (*(x + i) < minDist) {
            minDist = *(x + i);
            minIndex = i;
        }
    }

    return std::make_pair(minIndex, minDist);
}

void cdist(const Point& point, const std::vector<Point>& centers, int nClusters,
           double* distances, std::string metric) {
    for (int i = 0; i < nClusters; ++i) {
        distances[i] = dist_f(metric)(point, centers[i]);
    }
}

