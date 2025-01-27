#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

class LinearRegression {
private:
    vector<vector <double> > X;
    vector<double> Y;
    vector<double> w;
    double b;

    double dot(const vector<double> A, const vector<double> B) {
        int n = A.size();
        int m = B.size();

        if (n != m) {
            throw invalid_argument("The dimensions of the vectors are not compatible for dot product");
        }

        double result = 0;

        for (int i = 0; i < n; i++) {
            result += A[i] * B[i];
        }

        return result;
    }

    double computeCost() {
        int m = X.size();
        int featureCount = X[0].size();

        double cost = 0;
        for (int i = 0; i < m; i++)
        {
            double f_wb = dot(w, X[i]) + b;
            cost += pow(f_wb - Y[i], 2);
        }
        cost /= 2*m;

        return cost;

    }

    vector<double> computeGradientW() {
        int m = X.size();
        int featureCount = X[0].size();
        vector<double> dw(featureCount, 0);

        for (int i = 0; i < m; i++)
        {
            double error = dot(w, X[i]) + b - Y[i];
            for (int j = 0; j < featureCount; j++)
            {
                dw[j] += error * X[i][j];
            }
        }
        for (int j = 0; j < featureCount; j++)
        {
            dw[j] /= m;
        }
        return dw;
    }

    double computeGradientB() {
        int m = X.size();
        int featureCount = X[0].size();
        double db = 0;

        for (int i = 0; i < m; i++)
        {
            double f_wb = dot(w, X[i]) + b;
            db += f_wb - Y[i];
        }
        return db/m;
    }

public:
    LinearRegression(const vector<vector<double> >& x, const vector<double>& y) : X(x), Y(y), w(x[0].size(),0.5), b(0) {}

    void fit(double alpha, double iterations) {
        int featureCount = X[0].size();

        for (int i = 0; i < iterations; i++)
        {
            vector<double> dw = computeGradientW();
            double db = computeGradientB();

            for (int j = 0; j < featureCount; j++)
            {
                w[j] = w[j] - alpha * dw[j];
            }
            b = b - alpha * db;
            if (i % 100 == 0) {
                cout << "Iteration: " << i << " Cost: " << computeCost() << endl;
            }
        }
        
    }

    double predict(vector<double> x) {
        return dot(w, x) + b;
    }

    void printParameters() {
        cout << "Weights:";
        for (int i = 0; i < w.size(); i++) {
            cout << w[i] << " ";
        }
        cout << endl;
        cout << "Bias:" << b << endl;
        
    }
};

int main() {
    vector<vector<double> > X = {{1, 2}, {2,4}, {3, 6}, {4,8}, {5, 10}};
    vector<double> Y = {4, 7, 10, 13, 16};

    LinearRegression lr(X, Y);

    double learningRate = 0.01;
    double iterations = 10000;

    lr.fit(learningRate, iterations);
    lr.printParameters();

    vector<double> x = {6,12};
    cout << "Prediction for the given data" << ": " << lr.predict(x) << endl;

    return 0;
}