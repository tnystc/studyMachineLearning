#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

class LogisticRegression {
private:
    vector<vector <double> > X;
    vector<double> Y;
    vector<double> w;
    double b;
    vector<double> minValues, maxValues;
    //implementing dot product for two vectors
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

    //Normalization of the data
    void normalize() {
        int featureCount = X[0].size();
        minValues = vector<double>(featureCount, 0);
        maxValues = vector<double>(featureCount, 0);

        for (int i = 0; i < featureCount; i++) {
            minValues[i] = X[0][i];
            maxValues[i] = X[0][i];
        }

        for (int i = 1; i < X.size(); i++) {
            for (int j = 0; j < featureCount; j++) {
                if (X[i][j] < minValues[j]) {
                    minValues[j] = X[i][j];
                }
                if (X[i][j] > maxValues[j]) {
                    maxValues[j] = X[i][j];
                }
            }
        }

        for (int i = 0; i < X.size(); i++) {
            for (int j = 0; j < featureCount; j++) {
                X[i][j] = (X[i][j] - minValues[j]) / (maxValues[j] - minValues[j]);
            }
        }
    }

    double sigmoid(double z) {
        return 1 / (1 + exp(-z)); 
    }

    //cost function for logistic regression
    double computeCost() {
        int m = X.size();
        int featureCount = X[0].size();

        double cost = 0;
        
        for (int i = 0; i < m; i++){
            double f_wb = dot(w, X[i]) + b;
            cost += Y[i] * log(sigmoid(f_wb)) + (1 - Y[i]) * log(1-sigmoid(f_wb));
        }
        cost /= -m;
        return cost;

    }

    //derivative of cost function with respect to weight
    vector<double> computeGradientW() {
        int m = X.size();
        int featureCount = X[0].size();
        vector<double> dw(featureCount, 0);
        
        for (int i = 0; i < m; i++){
            double f_wb = dot(w, X[i]) + b;
            for (int j = 0; j < featureCount; j++){
                dw[j] += (sigmoid(f_wb) - Y[i]) * X[i][j];
            }
        }
        for (int j = 0; j < featureCount; j++){
            dw[j] /= m;
        }

        return dw;
    }

    //derivative of cost function with respect to bias
    double computeGradientB() {
        int m = X.size();
        int featureCount = X[0].size();
        double db = 0;
        
        for (int i = 0; i < m; i++){
            double f_wb = dot(w, X[i]) + b;
            db += sigmoid(f_wb) - Y[i];
        }

        return db/m;
    }

public:
    LogisticRegression(const vector<vector<double> >& x, const vector<double>& y) : X(x), Y(y), w(x[0].size(),0.5), b(0) {
        normalize();
    }

    //gradient decent algorithm
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
        for (int i = 0; i < x.size(); i++) {
            x[i] = (x[i] - minValues[i]) / (maxValues[i] - minValues[i]);
        }
        return sigmoid(dot(w, x) + b);
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
    // Binary classification data: study hours, past grades -> pass(1) / fail(0)
    vector<vector<double>> X = {{2, 50}, {3, 55}, {5, 60}, {7, 65}, {8, 70}, {10, 80}, {12, 85}, {15, 90}};
    vector<double> Y = {0, 0, 0, 1, 1, 1, 1, 1};

    LogisticRegression lr(X, Y);

    double learningRate = 0.2;
    double iterations = 10000;

    lr.fit(learningRate, iterations);
    lr.printParameters();

    vector<double> x = {3,70};
    double prediction = lr.predict(x);
    cout << "Prediction probability: " << prediction << endl;
    cout << "Prediction: " << (prediction >= 0.5 ? 1 : 0) << endl;
    return 0;
}