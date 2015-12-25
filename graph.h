#ifndef __GRAPH_H_
#define __GRAPH_H_

#include <iostream>
#include <vector>
#include <fstream>
using namespace std;

//node of graph
struct Node{
    int data;
    double pp;
    vector<int> out_node;
    vector<int> in_node;
    vector<double> edge_weight;
};

//
struct PNode{
    int index;          // the number of node
    double inc_inf_spd;    // the incremental influence spread based on act_sets
    PNode(int x, double y){
        index = x;
        inc_inf_spd = y;
    }
};

class Graph{
public:
    vector<Node> graph;
    Graph(int max_size){
        //graph = new vector<Node>();
        graph.resize(max_size);
    }

    void read_graph(string file_name, double p){
        ifstream ifs(file_name.c_str(), ios::in);
        int x, y;
        while(ifs>>x>>y){
            graph[x].out_node.push_back(y);
            graph[y].in_node.push_back(x);
            graph[x].edge_weight.push_back(p);
        }
        ifs.close();
    }

	void set_weight(){
		int num_node = graph.size();
		double len = 0;
		vector<int> edgew;
		for(int i = 0; i < num_node; ++i){
			len = (double)(graph[i].in_node.size());
			graph[i].pp = 1.0 / len;
		}
        for(int i = 0; i < num_node; ++i){
            edgew = graph[i].out_node;
            for(int j = 0; j < edgew.size(); ++j){
                graph[i].edge_weight[j] = graph[edgew[j]].pp;
            }
        }
	}
};

#endif // __GRAPH_H_
