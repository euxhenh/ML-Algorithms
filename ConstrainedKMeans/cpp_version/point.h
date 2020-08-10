#pragma once

#ifndef POINT_H
#define POINT_H

#include <assert.h>

#include <climits>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

class Point {
   private:
   public:
    int dim;
    double* coordinates;
    Point(){};
    Point(int nFeatures);
    Point(int fillValue, int nFeatures);
    Point(double* coords, int nFeatures);
    Point(const Point&);
    ~Point();
    Point& operator=(Point&& a);
    double sum();
    double min();
    double max();
    void fill(double);
};

Point operator+(const Point&, const Point&);
Point& operator+=(Point&, const Point&);
Point operator-(const Point&, const Point&);
Point& operator-=(Point&, const Point&);
Point operator*(const Point&, const Point&);
Point& operator*=(Point&, const Point&);
Point operator/(const Point&, const Point&);
Point& operator/=(Point&, const Point&);

Point operator+(const Point&, double);
Point& operator+=(Point&, double);
Point operator-(const Point&, double);
Point& operator-=(Point&, double);
Point operator*(const Point&, double);
Point& operator*=(Point&, double);
Point operator/(const Point&, double);
Point& operator/=(Point&, double);

std::ostream& operator<<(std::ostream&, const Point&);

Point point_pow(const Point&, int);

#endif
