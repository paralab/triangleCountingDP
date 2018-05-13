//
// Created by nishith on 1/23/18.
//
#include <vector>
#include <string>
#include "graph.h"
//using namespace std;

#ifndef TRIANGLE_COUNT_UTILS_H
#define TRIANGLE_COUNT_UTILS_H

// This function gets the maximum value from a vector of vectors of type integers >= 0
template <typename T>
T getMax2d(std::vector<edge<T> > inp){
    T maxVal = 0;
    for(T x = 0; x < inp.size(); x++)
    {
        if(inp[x].u > maxVal)
            maxVal = inp[x].u;
        if(inp[x].v > maxVal)
            maxVal = inp[x].v;
    }
    return maxVal;
}

// This function will go through iterators and output the first value where edge.u is difference from first->edge.u
//template <typename T>
//typename vector<edge<T>>::iterator findNext(vector<edge<T>>::iterator first, vector<edge<T>>::iterator last)
//{
//    vector<edge<T>>::iterator inc = first;
//    while (inc!=last) {
//        if (inc->u != first->u) return inc;
//        ++inc;
//    }
//    return last;
//}

template <typename T>
class quadTreeNode{
public:
    T **buckets;
    T umask, vmask;
    std::vector<quadTreeNode<T>*> children;
    int base;
    quadTreeNode(int bucketBase){
        base = bucketBase;
        buckets = new T*[bucketBase];
        for(int i = 0; i < bucketBase; i++)
            buckets[i] = new T[bucketBase]();
    }
    quadTreeNode(){
        base = 16;
        buckets = new T*[16];
        for(int i = 0; i < 16; i++)
            buckets[i] = new T[16]();
    }
    std::string repr() {
        std::string out;
        out += "(" + std::to_string(umask) + "," + to_string(vmask) + ")\n";
        for (int i = 0; i < base; i++) {
            for (int j = 0; j < base; j++) {
                out += to_string(buckets[i][j]);
                if(buckets[i][j]) {
                    out += "(";
                    out += std::to_string(i);
                    out += ",";
                    out += std::to_string(j);
                    out += ")";
                }
                out += "\t\t";
            }
            out+="\n";
        }
        return out;
    }
};

template <typename T>
quadTreeNode<T>* bucketSixteenEdges(std::vector<edge<T> > inp, T level, T umask, T vmask, bool isRoot) {
    if(level < 0)
        return NULL;
    quadTreeNode<T> *node= new quadTreeNode<T>;
    //T buckets[256] = {};
    node->umask = umask;
    node->vmask = vmask;
    node->children.clear();
    for (int i = 0; i < inp.size(); i++) {
        //node.buckets[(inp[i].u >> (level * 4) & 0b1111) << 4 | (inp[i].v >> (level * 4) & 0b1111)]++;
        if(isRoot || ((inp[i].u >> ((level+1) * 4) & umask) && (inp[i].v >> ((level+1) * 4) & vmask)))
            node->buckets[(inp[i].u >> (level * 4) & 0b1111)][(inp[i].v >> (level * 4) & 0b1111)]++;
    }
    std::cout<<level<<std::endl;
    std::cout<<node->repr();
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            /*
            if(node.buckets[i << 4 | j])
                cout << node.buckets[i << 4 | j] << "("<<(i<<4|j)<<") ";
            else
                cout << node.buckets[i << 4 | j] << " ";
            */
            if(node->buckets[i][j]) {
                //cout << node->buckets[i][j] << "(" << i << "," << j << ") ";
                quadTreeNode<T> *child = bucketSixteenEdges(inp, level - 1, i, j, false);
                if(child!=NULL)
                    node->children.push_back(child);
            }
        }
        //cout << endl;
    }
    return node;
}

template <typename T>
T double_buffer_radixsort_twoArrays_trianglecount(edge<T> *E, edge<T> *EPrime, T Esize, T EPrimesize, int bitnum){
    // count
    T bucketsE[4] = {};
    T bucketsEPrime[4] = {};
    T bucketsEScan[4] = {};
    T bucketsEPrimeScan[4] = {};
    T bucketsEScancpy[4] = {};
    T bucketsEPrimeScancpy[4] = {};
    for(T i = 0; i < Esize; i++){
        bucketsE[((E[i].u>>bitnum)&0b1)<<1 | ((E[i].v>>bitnum)&0b1)]++;
    }
    for(T i = 0; i < EPrimesize; i++){
        bucketsEPrime[((EPrime[i].u>>bitnum)&0b1)<<1 | ((EPrime[i].v>>bitnum)&0b1)]++;
    }

    // scan each
    for(int i = 1; i < 4; i++){
        bucketsEScan[i] = bucketsE[i-1]+bucketsEScan[i-1];
        bucketsEScancpy[i] = bucketsEScan[i];
        bucketsEPrimeScan[i] = bucketsEPrime[i-1]+bucketsEPrimeScan[i-1];
        bucketsEPrimeScancpy[i] = bucketsEPrimeScan[i];
    }

    // move for conditions
    std::vector<edge<T> > eb, epb;
    eb.reserve(Esize);
    epb.reserve(EPrimesize);
    edge<T> *ebuf = eb.data();
    edge<T> *epbuf = epb.data();
    // move data into the relevant buckets
    for(T i = 0; i < Esize; i++){
        int f = ((E[i].u>>bitnum)&0b1)<<1 | ((E[i].v>>bitnum)&0b1);
        ebuf[bucketsEScan[f]].u = E[i].u;
        ebuf[bucketsEScan[f]].v = E[i].v;
        bucketsEScan[f]++;
    }
    for(T i = 0; i < EPrimesize; i++){
        int f = ((EPrime[i].u>>bitnum)&0b1)<<1 | ((EPrime[i].v>>bitnum)&0b1);
        epbuf[bucketsEPrimeScan[f]].u = EPrime[i].u;
        epbuf[bucketsEPrimeScan[f]].v = EPrime[i].v;
        bucketsEPrimeScan[f]++;
    }
    // copy data into the original containers Ein.swap(eb); EPrimein.swap(epb); and destroy temp data
    for(T i = 0; i < Esize; i++)
        E[i] = ebuf[i];
    for(T i = 0; i < EPrimesize; i++)
        EPrime[i] = epbuf[i];
    std::vector<edge<T> >().swap(eb);
    std::vector<edge<T> >().swap(epb);

    T trianglecount = 0;
    // recurse only if both buckets are not 0
    for(int i = 0; i < 4; i++){
        // if either bucket has no elements do not recurse
        if(bucketsE[i]==0 || bucketsEPrime[i]==0)
            continue;
        // if either buckets have a single element perform a linear search for the element in the other and
        // increment the number of times its found in EPrime
        if(bucketsE[i]==1){
            T start = bucketsEPrimeScancpy[i];
            T end = bucketsEPrimeScan[i];
            for(T j = start; j < end; j++){
                if(EPrime[j] == E[bucketsEScancpy[i]])
                    trianglecount++;
            }
            continue;
        }
        if(bucketsEPrime[i]==1){
            T start = bucketsEScancpy[i];
            T end= bucketsEScan[i];
            for(T j = start; j < end; j++){
                if(E[j] == EPrime[bucketsEPrimeScancpy[i]])
                    trianglecount++;
            }
            continue;
        }
        // If the counts are different, recurse
        if(bitnum > 0) {
            trianglecount += double_buffer_radixsort_twoArrays_trianglecount(E + bucketsEScancpy[i],
                                                                             EPrime + bucketsEPrimeScancpy[i], bucketsE[i],
                                                                             bucketsEPrime[i], bitnum - 1);
        }else{
            // If you are on the last bit that means that the EPrime count is what should be added
            // The loop should not reach here since if the bits are effectively the same, they should be counted in the
            // case of bucketsE[i] = 1 and bucketsEPrime[i] = n
            trianglecount += bucketsEPrime[i];
            std::cout<<"THIS CASE SHOULD NOT OCCUR: RESOLUTION ON LAST BIT HAPPENNED - PLEASE INVESTIGATE."<<std::endl;
        }
    }
    return trianglecount;
}
#endif //TRIANGLE_COUNT_UTILS_H
