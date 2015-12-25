#ifndef __SOLUTION_H_
#define __SOLUTION_H_

//#include <vector>
#include <stack>
#include <queue>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include "graph.h"



//using namespace std;

vector<vector<double> > sps;

bool operator < (const PNode &nd1, const PNode &nd2){
    return (nd1.inc_inf_spd < nd2.inc_inf_spd);
}

class Solution{
private:
    //the number of influence node under the seed set act_sets
    int one_diffusion_wc(const Graph &graph, const vector<int> &act_sets, const double &p){
        int num_active = act_sets.size();
        int num_graph = graph.graph.size();
        vector<bool> flags(num_graph, false);
        stack<int> tstk;
        int num_act_sets = act_sets.size();
		double xpp = 0.0;
        for(int i = 0; i < num_act_sets; ++i){
            tstk.push(act_sets[i]);
            flags[act_sets[i]] = true;
        }
        while(!tstk.empty()){
            int cur = tstk.top();
            tstk.pop();
            vector<int> cur_out_node = graph.graph[cur].out_node;
            int cur_out_size = cur_out_node.size();
            for(int i = 0; i < cur_out_size; ++i){
                double cur_p = rand() / (double)(RAND_MAX);
				xpp = graph.graph[cur_out_node[i]].pp;
                if(flags[cur_out_node[i]] == false && cur_p <= xpp){ //xpp->p in IC model
                    tstk.push(cur_out_node[i]);
                    num_active += 1;
                    flags[cur_out_node[i]] = true;
                }
            }
        }
        return num_active;
    }

	//the number of influence node under the seed set act_sets
    int one_diffusion_ic(const Graph &graph, const vector<int> &act_sets, const double &p){
        int num_active = act_sets.size();
        int num_graph = graph.graph.size();
        vector<bool> flags(num_graph, false);
        stack<int> tstk;
        int num_act_sets = act_sets.size();
		double xpp = 0.0;
        for(int i = 0; i < num_act_sets; ++i){
            tstk.push(act_sets[i]);
            flags[act_sets[i]] = true;
        }
        while(!tstk.empty()){
            int cur = tstk.top();
            tstk.pop();
            vector<int> cur_out_node = graph.graph[cur].out_node;
            int cur_out_size = cur_out_node.size();
            for(int i = 0; i < cur_out_size; ++i){
                double cur_p = rand() / (double)(RAND_MAX);
                if(flags[cur_out_node[i]] == false && cur_p <= p){ //xpp->p in IC model
                    tstk.push(cur_out_node[i]);
                    num_active += 1;
                    flags[cur_out_node[i]] = true;
                }
            }
        }
        return num_active;
    }

    //set vector<bool> false
    void set_vector_false(vector<bool> &flags){
        for(unsigned int i = 0; i < flags.size(); ++i){
            flags[i] = false;
        }
    }

    //get error between vector1 and vector2
    double get_error_two_vec(vector<double> &prs, vector<double> &pre_prs){
        int num_node = prs.size();
        double error_two_vec = 0.0;
        for(int i = 0; i < num_node; ++i){
            error_two_vec += (prs[i] - pre_prs[i])*(prs[i] - pre_prs[i]);
            pre_prs[i] = prs[i];
        }
        return error_two_vec;
    }

    //get the page rank value of each node
    void get_page_ranks_node(const Graph &graph, vector<double> &prs, const double &error_convergence, const int &max_ite, const double &d){
        double cur_error = 1.0;
        int cur_ite = 0, num_node = graph.graph.size();
        vector<double> pre_pres(num_node, 0.0);
        while(cur_ite < max_ite && cur_error > error_convergence){
            for(int i = 0; i < num_node; ++i){
                double tmp_res = 0.0;
                vector<int> innode = graph.graph[i].in_node;
				int len_in = innode.size();
                for(int j = 0; j < len_in; ++j){
                    tmp_res += prs[innode[j]] / (double)(graph.graph[innode[j]].out_node.size());
                }
                prs[i] = (1 - d) + d * tmp_res;
				//prs[i] = (1 - d) + d * tmp_res / (double)(len_in);
            }
            cur_ite += 1;
            cur_error = get_error_two_vec(prs, pre_pres);
        }
    }

    //get_most_inf_max_sets
    void get_most_inf_max_sets(const vector<double> &prs, vector<int> &ans, const int &K){
        vector<double> max_val(K, 0.0);
        vector<int> max_index(K, 0);
        double min_val = 0;
        int min_index = 0;
        for(int i = 0; i < K; ++i){
            max_val[i] = prs[i];
            max_index[i] = i;
        }
        for(unsigned int i = K; i < prs.size(); ++i){
            min_val = max_val[0];
            min_index = 0;
            for(int j = 1; j < K; ++j){
                if(min_val > max_val[j]){
                    min_val = max_val[j];
                    min_index = j;
                }
            }
            if(min_val < prs[i]){
                max_val[min_index] = prs[i];
                max_index[min_index] = i;
                //cout<<"######"<<endl;
            }
        }
        for(int i = 0; i < K; ++i){
            ans.push_back(max_index[i]);
        }
    }

    // get random k nodes from num_node nodes
    void get_random_k(const int &num_node, vector<int> &ans, const int &K){
        vector<int> idx(num_node, 0);
        int rdm = 0;
        for(int i = 0; i < num_node; ++i)
            idx[i] = i;
        for(int i = 0; i < K; ++i){
            rdm = rand() % (num_node - i);
            ans.push_back(idx[rdm]);
            idx[rdm] = idx[num_node - 1 - i];
        }
    }

    // get top k node based on degree discount
    void get_topk_degree_discount(const Graph &graph, const vector<double> &degrees, vector<int> &ans, const int &K, const double &p){
        int num_node = degrees.size(), olen = 0;
        vector<bool> flags(num_node, false);
        vector<double> tvs(num_node, 0.0);
        vector<double> dds(num_node, 0.0);
		vector<int> onodes;
        int max_index = 0;
        double max_val = 0.0;
        for(int i = 0; i < num_node; ++i)
            dds[i] = degrees[i];
        for(int i = 0; i < K; ++i){
            max_index = 0;
            max_val = 0.0;
            for(int j = 0; j < num_node; ++j){
                if(flags[j] == false && max_val <= dds[j]){
                    max_index = j;
                    max_val = dds[j];
                }
            }
            ans.push_back(max_index);
            flags[max_index] = true;
			onodes = graph.graph[max_index].out_node;
			olen = onodes.size();
            for(int j = 0; j < olen; ++j){
                if(flags[onodes[j]] == false){
                    tvs[onodes[j]] += 1;
                    dds[onodes[j]] = degrees[onodes[j]] - 2 * tvs[onodes[j]] - (degrees[onodes[j]] - tvs[onodes[j]]) * tvs[onodes[j]] * p;
                }
            }

            onodes = graph.graph[max_index].in_node;
			olen = onodes.size();
            for(int j = 0; j < olen; ++j){
                if(flags[onodes[j]] == false){
                    tvs[onodes[j]] += 1;
                    dds[onodes[j]] = degrees[onodes[j]] - 2 * tvs[onodes[j]] - (degrees[onodes[j]] - tvs[onodes[j]]) * tvs[onodes[j]] * p;
                }
            }
        }
    }

    // set distance maximum
    void set_distance_maximum(const double &max_val, vector<double> &min_distance){
        for(unsigned int i = 0; i < min_distance.size(); ++i){
            min_distance[i] = max_val;
        }
    }

    // get index of minimum from min_distance
    int get_index_min_distance(const vector<double> &min_distance, const vector<bool> &flags){
        int min_index = 0, num_node = flags.size();
        double min_val = 0.0;
        for(int i = 0; i < num_node; ++i){
            if(flags[i] == false){
                min_val = min_distance[i];
                min_index = i;
                break;
            }
        }
        for(int i = 0; i < num_node; ++i){
            if(flags[i] == false && min_val > min_distance[i]){
                min_val = min_distance[i];
                min_index = i;
            }
        }
        return min_index;
    }

    // calculate single source shortest path
    void calc_sssp_dijstra(const Graph &graph, const int &source, vector<double> &min_distance){
        int num_node =graph.graph.size(), k = 0;
        vector<bool> flags(num_node, false);
        min_distance[source] = 0.0;
        while(k < num_node){
            int cur_node = get_index_min_distance(min_distance, flags);
            vector<int> cur_outnode = graph.graph[cur_node].out_node;
            for(unsigned int i = 0; i < cur_outnode.size(); ++i){
                //cout<<cur_node<<' '<<cur_outnode[i]<<"#####"<<endl;
                if(min_distance[cur_outnode[i]] > min_distance[cur_node] - log10(graph.graph[cur_node].edge_weight[i])){
                    min_distance[cur_outnode[i]] = min_distance[cur_node] - log10(graph.graph[cur_node].edge_weight[i]);
                }
            }
            //cout<<cur_node<<endl;
            flags[cur_node] = true;
            k += 1;
        }
    }

    // calculate all shortest path between two nodes
    void calc_sps_dijstra(const Graph &graph){
        int num_node = graph.graph.size();
        sps.resize(num_node);
        for(int i = 0; i < num_node; ++i){
            sps[i] = vector<double>(num_node, DBL_MAX);
        }
        for(int i = 0; i < num_node; ++i){
            calc_sssp_dijstra(graph, i, sps[i]);
        }
    }

    // calculate all shortest path between two nodes
    void calc_sps_dijstra_p(const Graph &graph){
        int num_node = graph.graph.size(), i = 0;
        sps.resize(num_node);
        for(int i = 0; i < num_node; ++i){
            sps[i] = vector<double>(num_node, DBL_MAX);
        }
        #pragma omp parallel shared(graph, sps) private(i)
        {
            #pragma omp for schedule(static)
            for(i = 0; i < num_node; ++i){
                calc_sssp_dijstra(graph, i, sps[i]);
            }
        }
    }

    // iteratively calculate the incrmental influence of each node
    void calc_irie_iteration(const Graph &graph, const vector<double> &aps, const vector<bool> &flags, const double &p, const double &alpha, vector<double> &infs){
        double converg_error = 0.01, cur_error = 1.0;
        int max_iter = 100, cur_iter = 0, num_node = graph.graph.size();
        vector<double> pre_infs(num_node, 0.0);
        for(int i = 0; i < num_node; ++i)
            pre_infs[i] = infs[i];
        while(cur_error > converg_error && max_iter > cur_iter){
            for(int i = 0; i < num_node; ++i){
                if(flags[i] == false){
                    vector<int> outnode = graph.graph[i].out_node;
                    double tmp_d = 0.0;
                    for(unsigned int j = 0; j < outnode.size(); ++j){
                        tmp_d += (1 - aps[i]) * (1 + alpha * p * infs[outnode[j]]);
                    }
                    infs[i] = tmp_d;
                }
            }
            cur_error = get_error_two_vec(infs, pre_infs);
            cur_iter += 1;
        }
    }

    // get the most influential inactive node
    int get_cur_top_index(const vector<bool> &flags, const vector<double> &infs){
        int max_index = 0, num_node = infs.size();
        double max_val = 0.0;
        for(int i = 0; i < num_node; ++i){
            if(flags[i] == false){
                max_index = i;
                max_val = infs[i];
                break;
            }
        }
        for(int i = 0; i < num_node; ++i){
            if(flags[i] == false && infs[i] > max_val){
                max_index = i;
                max_val = infs[i];
            }
        }
        //cout<<max_index<<' '<<max_val<<endl;
        return max_index;
    }

public:

    //get the accurate the results by monte carlo under weight cascade model
    double monte_carlo_diffusions_wc(const Graph &graph, const vector<int> &act_sets, const double &p, const int &R){
        int active_num = 0;
        for(int i = 0; i < R; ++i){
            active_num += one_diffusion_wc(graph, act_sets, p);
        }
        return ((double)(active_num) / (double)(R));
    }

	//get the accurate the results by monte carlo
    double monte_carlo_diffusions_ic(const Graph &graph, const vector<int> &act_sets, const double &p, const int &R){
        int active_num = 0;
        for(int i = 0; i < R; ++i){
            active_num += one_diffusion_ic(graph, act_sets, p);
        }
        return ((double)(active_num) / (double)(R));
    }

    //get the accurate the results by monte carlo
    double monte_carlo_diffusions_wc_parallel(const Graph &graph, const vector<int> &act_sets, const double &p, const int &R, const int &num_thread){
        int active_num = 0, i = 0;
        vector<int> sum(num_thread, 0);
        cout<<num_thread<<endl;
        #pragma omp parallel shared(graph, act_sets, p) private(i)
        {
            int tid = omp_get_thread_num();
            srand((unsigned)time(NULL) + tid);
            #pragma omp for schedule(static)
            for(i = 0; i < R; ++i){
                sum[tid] += one_diffusion_wc(graph, act_sets, p);
            }
        }
        int len_sum = sum.size();
        for(i = 0; i < len_sum; ++i){
            active_num += sum[i];
            cout<<sum[i]<<endl;
        }
        return ((double)(active_num) / (double)(R));
    }

    //kemple greedy algorithm
    vector<int> greedy_wc(const Graph &graph, const double &p, const int &R, const int &K){
        vector<int> ans;
        int num_node = graph.graph.size(), cur_node = 0;
        vector<bool> flags(num_node, false);
        double cur_max = 0.0;
        for(int k = 0; k < K; ++k){
            cur_max = 0.0;
            cur_node = 0;
            for(int i = 0; i < num_node; ++i){
                if(flags[i] == false){
                    ans.push_back(i);
                    double cur_val = monte_carlo_diffusions_wc(graph, ans, p, R);
                    if(cur_val > cur_max){
                        cur_node = i;
                        cur_max = cur_val;
                    }
                    ans.pop_back();
                }
            }
            ans.push_back(cur_node);
            flags[cur_node] = true;
			cout<<cur_max<<endl;
        }
        return ans;
    }

    //celf algorithm, 700th faster than greedy
    vector<int> celf_wc(const Graph &graph, const double &p, const int &R, const int &K){
        vector<int> ans;
        int num_node = graph.graph.size(), cur_node = 0;
        vector<bool> flags(num_node, false);
        priority_queue<PNode> pre_queue;
        double cur_influence_spread = 0.0;
        for(int i = 0; i < num_node; ++i){
            pre_queue.push(PNode(i, (double)(num_node) + 1.0));
        }
        while(ans.size() < K){
            set_vector_false(flags);
            cur_influence_spread = monte_carlo_diffusions_wc(graph, ans, p, R);
            while(true){
                if(flags[pre_queue.top().index]){
                    ans.push_back(pre_queue.top().index);
                    pre_queue.pop();
                    break;
                }else{
                    cur_node = pre_queue.top().index;
                    pre_queue.pop();
                    ans.push_back(cur_node);
                    pre_queue.push(PNode(cur_node, monte_carlo_diffusions_wc(graph, ans, p, R) - cur_influence_spread));
                    ans.pop_back();
                    flags[cur_node] = true;
                }
            }
			cout<<pre_queue.top().inc_inf_spd<<endl;
        }
        //cout<<monte_carlo_diffusions_wc(graph, ans, p, R)<<endl;
        return ans;
    }

	//celf algorithm, 700th faster than greedy
    vector<int> celf_ic(const Graph &graph, const double &p, const int &R, const int &K){
        vector<int> ans;
        int num_node = graph.graph.size(), cur_node = 0;
        vector<bool> flags(num_node, false);
        priority_queue<PNode> pre_queue;
        double cur_influence_spread = 0.0;
        for(int i = 0; i < num_node; ++i){
            pre_queue.push(PNode(i, (double)(num_node) + 1.0));
        }
        while(ans.size() < K){
            set_vector_false(flags);
            cur_influence_spread = monte_carlo_diffusions_ic(graph, ans, p, R);
            while(true){
                if(flags[pre_queue.top().index]){
                    ans.push_back(pre_queue.top().index);
                    pre_queue.pop();
                    break;
                }else{
                    cur_node = pre_queue.top().index;
                    pre_queue.pop();
                    ans.push_back(cur_node);
                    pre_queue.push(PNode(cur_node, monte_carlo_diffusions_ic(graph, ans, p, R) - cur_influence_spread));
                    ans.pop_back();
                    flags[cur_node] = true;
                }
            }
			cout<<pre_queue.top().inc_inf_spd<<endl;
        }
        //cout<<monte_carlo_diffusions_ic(graph, ans, p, R)<<endl;
        return ans;
    }

    //random get K sets
    vector<int> random(const Graph &graph, const int &K){
        vector<int> ans;
        int num_node = graph.graph.size();
        get_random_k(num_node, ans, K);
        cout<<monte_carlo_diffusions_wc(graph, ans, 0.01, 10000)<<endl;
        return ans;
    }

    //out degree based algorithm
    vector<int> degree_out(const Graph &graph, const int &K){
        vector<int> ans;
        int num_node = graph.graph.size();
        vector<double> degrees(num_node, 0.0);
        for(int i = 0; i < num_node; ++i){
            degrees[i] = (double)(graph.graph[i].out_node.size());
        }
        get_most_inf_max_sets(degrees, ans, K);
        cout<<monte_carlo_diffusions_wc(graph, ans, 0.01, 10000)<<endl;
        return ans;
    }

	//degree based algorithm
    vector<int> degree(const Graph &graph, const int &K){
        vector<int> ans;
        int num_node = graph.graph.size();
        vector<double> degrees(num_node, 0.0);
        for(int i = 0; i < num_node; ++i){
            degrees[i] = (double)(graph.graph[i].out_node.size() + graph.graph[i].in_node.size());
        }
        get_most_inf_max_sets(degrees, ans, K);
        cout<<monte_carlo_diffusions_wc(graph, ans, 0.01, 10000)<<endl;
        return ans;
    }

    // greedy methods based on candidate sets
    vector<int> cim_greedy(const Graph &graph, const double &p, const int &R, const int &K, const double &lamda){
        vector<int> cand_sets = degree(graph, (int)(lamda * K));
        vector<int> ans;
        int num_node = graph.graph.size(), num_cand = cand_sets.size(), cur_node = 0;
        vector<bool> flags(num_node, false);
        double cur_max = 0.0;
        for(int k = 0; k < K; ++k){
            cur_max = 0.0;
            cur_node = 0;
            for(int i = 0; i < num_cand; ++i){
                if(flags[cand_sets[i]] == false){
                    ans.push_back(cand_sets[i]);
                    double cur_val = monte_carlo_diffusions_wc(graph, ans, p, R);
                    if(cur_val > cur_max){
                        cur_node = cand_sets[i];
                        cur_max = cur_val;
                    }
                    ans.pop_back();
                }
            }
            ans.push_back(cur_node);
            flags[cur_node] = true;
        }
        return ans;
    }

    // celf methods based on candidate sets
    vector<int> cim_celf(const Graph &graph, const double &p, const int &R, const int &K, const double &lamda){
        vector<int> cand_sets = degree(graph, (int)(lamda * K));
        vector<int> ans;
        int num_node = graph.graph.size(), num_cand = cand_sets.size(), cur_node = 0, xcnt= 0;
        vector<bool> flags(num_node, false);
        priority_queue<PNode> pre_queue;
        double cur_influence_spread = 0.0;
        for(int i = 0; i < num_cand; ++i){
            pre_queue.push(PNode(cand_sets[i], (double)(num_node) + 1.0));
        }
        while(ans.size() < K){
            set_vector_false(flags);
            cur_influence_spread = monte_carlo_diffusions_wc(graph, ans, p, R);
            while(true){
                if(flags[pre_queue.top().index]){
                    ans.push_back(pre_queue.top().index);
                    cout<<xcnt<<endl;
                    xcnt = 0;
                    pre_queue.pop();
                    break;
                }else{
                    cur_node = pre_queue.top().index;
                    pre_queue.pop();
                    ans.push_back(cur_node);
                    pre_queue.push(PNode(cur_node, monte_carlo_diffusions_wc(graph, ans, p, R) - cur_influence_spread));
                    ans.pop_back();
                    flags[cur_node] = true;
                    xcnt += 1;
                }
            }
        }
        //cout<<monte_carlo_diffusions_wc(graph, ans, p, R)<<endl;
        return ans;
    }

};

#endif // __SOLUTION_H_
