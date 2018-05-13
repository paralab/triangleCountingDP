//
// Created by nishith on 1/23/18.
//
//#include <iostream>

#ifndef TRIANGLE_COUNT_GRAPH_H
#define TRIANGLE_COUNT_GRAPH_H
template <typename T>
class edge{
public:
    T v,u;

    edge(T v1, T v2) :
    u(v1), v(v2)
    { }

    // Comparision operator
    static bool comp(edge a, edge b){
        if(a.u < b.u) {
            return true;
        }
	if(a.u > b.u){
            return false;
        }
	if(a.v < b.v){
            return true;
        }
	return false;
    }

    void print(){
        std::cout<<"("<<u<<", "<<v<<")";
    }

    // This function returns True if the u < v which means that its not double counted
    static bool isRepeatedEdge(edge a){
        return a.u > a.v;
    }

    inline bool operator == (const edge &Ref) const {
        return (this->u == Ref.u && this->v == Ref.v);
    }

    inline bool operator<(const edge &Ref) const {
        return edge<T>::comp(*this, Ref);
    }

    void operator = (const edge &Ref) {
        u = Ref.u;
        v = Ref.v;
    }
};

#endif //TRIANGLE_COUNT_GRAPH_H
