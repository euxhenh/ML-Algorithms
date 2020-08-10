#pragma once

#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "point.h"

double euclidean(const Point&, const Point&);

std::function<double(const Point&, const Point&)> dist_f(std::string);

std::pair<int, double> argmin(double*, int);

void cdist(const Point&, const std::vector<Point>&, int, double*, std::string);

#endif
