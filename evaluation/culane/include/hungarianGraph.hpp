#ifndef HUNGARIAN_GRAPH_HPP
#define HUNGARIAN_GRAPH_HPP
#include <vector>
using namespace std;

struct pipartiteGraph {
    vector<vector<double> > mat;
    vector<bool> leftUsed, rightUsed;
    vector<double> leftWeight, rightWeight;
    vector<int>rightMatch, leftMatch;
    int leftNum, rightNum;
    bool matchDfs(int u) {
        leftUsed[u] = true;
        for (int v = 0; v < rightNum; v++) {
            if (!rightUsed[v] && fabs(leftWeight[u] + rightWeight[v] - mat[u][v]) < 1e-2) {
                rightUsed[v] = true;
                if (rightMatch[v] == -1 || matchDfs(rightMatch[v])) {
                    rightMatch[v] = u;
                    leftMatch[u] = v;
                    return true;
                }
            }
        }
        return false;
    }
    void resize(int leftNum, int rightNum) {
        this->leftNum = leftNum;
        this->rightNum = rightNum;
        leftMatch.resize(leftNum);
        rightMatch.resize(rightNum);
        leftUsed.resize(leftNum);
        rightUsed.resize(rightNum);
        leftWeight.resize(leftNum);
        rightWeight.resize(rightNum);
        mat.resize(leftNum);
        for (int i = 0; i < leftNum; i++) mat[i].resize(rightNum);
    }
    void match() {
        for (int i = 0; i < leftNum; i++) leftMatch[i] = -1;
        for (int i = 0; i < rightNum; i++) rightMatch[i] = -1;
        for (int i = 0; i < rightNum; i++) rightWeight[i] = 0;
        for (int i = 0; i < leftNum; i++) {
            leftWeight[i] = -1e5;
            for (int j = 0; j < rightNum; j++) {
                if (leftWeight[i] < mat[i][j]) leftWeight[i] = mat[i][j];
            }
        }

        for (int u = 0; u < leftNum; u++) {
            while (1) {
                for (int i = 0; i < leftNum; i++) leftUsed[i] = false;
                for (int i = 0; i < rightNum; i++) rightUsed[i] = false;
                if (matchDfs(u)) break;
                double d = 1e10;
                for (int i = 0; i < leftNum; i++) {
                    if (leftUsed[i] ) {
                        for (int j = 0; j < rightNum; j++) {
                            if (!rightUsed[j]) d = min(d, leftWeight[i] + rightWeight[j] - mat[i][j]);
                        }
                    }
                }
                if (d == 1e10) return ;
                for (int i = 0; i < leftNum; i++) if (leftUsed[i]) leftWeight[i] -= d;
                for (int i = 0; i < rightNum; i++) if (rightUsed[i]) rightWeight[i] += d;
            }
        }
    }
};


#endif // HUNGARIAN_GRAPH_HPP
