#include "point.h"

/* ===== Constructors ===== */
Point::Point(int nFeatures) {
    dim = nFeatures;
    coordinates = new double[dim];
}

Point::Point(int fillValue, int nFeatures) {
    dim = nFeatures;
    coordinates = new double[dim];
    for (int i = 0; i < dim; ++i) coordinates[i] = fillValue;
}

Point::Point(double* coords, int nFeatures) {
    dim = nFeatures;
    coordinates = new double[dim];
    for (int i = 0; i < dim; ++i) coordinates[i] = coords[i];
}

Point::Point(const Point& p2) {
    dim = p2.dim;
    for (int i = 0; i < dim; ++i) coordinates[i] = p2.coordinates[i];
}

Point::~Point() { delete[] coordinates; }

Point& Point::operator=(Point&& a) {
    if (&a == this) return *this;

    //delete coordinates;

    coordinates = a.coordinates;
    dim = a.dim;
    a.coordinates = nullptr;

    return *this;
}

/* ===== Utils ===== */
double Point::sum() {
    double total = 0;
    for (int i = 0; i < dim; ++i) total += coordinates[i];
    return total;
}

double Point::min() {
    double min_value = INT_MAX;
    for (int i = 0; i < dim; ++i)
        min_value = std::min(min_value, coordinates[i]);
    return min_value;
}

double Point::max() {
    double max_value = INT_MIN;
    for (int i = 0; i < dim; ++i)
        max_value = std::max(max_value, coordinates[i]);
    return max_value;
}

void Point::fill(double fillValue) {
    for (int i = 0; i < dim; ++i) coordinates[i] = fillValue;
}

/* ====== Operators ====== */
Point operator+(const Point& p1, const Point& p2) {
    assert(p1.dim == p2.dim);
    Point point(p1.dim);
    for (int i = 0; i < p1.dim; ++i)
        point.coordinates[i] = p1.coordinates[i] + p2.coordinates[i];
    return point;
}

Point& operator+=(Point& p1, const Point& p2) {
    assert(p1.dim == p2.dim);
    for (int i = 0; i < p1.dim; ++i) p1.coordinates[i] += p2.coordinates[i];
    return p1;
}

Point operator-(const Point& p1, const Point& p2) {
    assert(p1.dim == p2.dim);
    Point point(p1.dim);
    for (int i = 0; i < p1.dim; ++i)
        point.coordinates[i] = p1.coordinates[i] - p2.coordinates[i];
    return point;
}

Point& operator-=(Point& p1, const Point& p2) {
    assert(p1.dim == p2.dim);
    for (int i = 0; i < p1.dim; ++i) p1.coordinates[i] -= p2.coordinates[i];
    return p1;
}

Point operator*(const Point& p1, const Point& p2) {
    assert(p1.dim == p2.dim);
    Point point(p1.dim);
    for (int i = 0; i < p1.dim; ++i)
        point.coordinates[i] = p1.coordinates[i] * p2.coordinates[i];
    return point;
}

Point& operator*=(Point& p1, const Point& p2) {
    assert(p1.dim == p2.dim);
    for (int i = 0; i < p1.dim; ++i) p1.coordinates[i] *= p2.coordinates[i];
    return p1;
}

Point operator/(const Point& p1, const Point& p2) {
    assert(p1.dim == p2.dim);
    Point point(p1.dim);
    for (int i = 0; i < p1.dim; ++i)
        point.coordinates[i] = p1.coordinates[i] / p2.coordinates[i];
    return point;
}

Point& operator/=(Point& p1, const Point& p2) {
    assert(p1.dim == p2.dim);
    for (int i = 0; i < p1.dim; ++i) p1.coordinates[i] /= p2.coordinates[i];
    return p1;
}

Point operator+(const Point& p1, double n) {
    Point point(p1.dim);
    for (int i = 0; i < p1.dim; ++i)
        point.coordinates[i] = p1.coordinates[i] + n;
    return point;
}

Point& operator+=(Point& p1, double n) {
    for (int i = 0; i < p1.dim; ++i) p1.coordinates[i] += n;
    return p1;
}

Point operator-(const Point& p1, double n) {
    Point point(p1.dim);
    for (int i = 0; i < p1.dim; ++i)
        point.coordinates[i] = p1.coordinates[i] - n;
    return point;
}

Point& operator-=(Point& p1, double n) {
    for (int i = 0; i < p1.dim; ++i) p1.coordinates[i] -= n;
    return p1;
}

Point operator*(const Point& p1, double n) {
    Point point(p1.dim);
    for (int i = 0; i < p1.dim; ++i)
        point.coordinates[i] = p1.coordinates[i] * n;
    return point;
}

Point& operator*=(Point& p1, double n) {
    for (int i = 0; i < p1.dim; ++i) p1.coordinates[i] *= n;
    return p1;
}

Point operator/(const Point& p1, double n) {
    if (n == 0) throw std::invalid_argument("Division by 0 encountered.");
    Point point(p1.dim);
    for (int i = 0; i < p1.dim; ++i)
        point.coordinates[i] = p1.coordinates[i] / n;
    return point;
}

Point& operator/=(Point& p1, double n) {
    if (n == 0) throw std::invalid_argument("Division by 0 encountered.");
    for (int i = 0; i < p1.dim; ++i) p1.coordinates[i] /= n;
    return p1;
}

std::ostream& operator<<(std::ostream& os, const Point& p) {
    for (int i = 0; i < std::min(p.dim, 10); ++i) os << p.coordinates[i] << " ";
    if (p.dim > 10) os << "...";
    os << std::endl;
    return os;
}

Point point_pow(const Point& p, int exp) {
    Point p_pow(p.dim);
    for (int i = 0; i < p.dim; ++i)
        p_pow.coordinates[i] = std::pow(p.coordinates[i], exp);
    return p_pow;
}
