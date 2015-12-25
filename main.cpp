//#include <iostream>
#include <time.h>
#include "graph.h"
#include "solution.h"

/*
# 6301, "./data/Gnu.txt"
# 15233, "./data/hep.txt"
# 8298, "./data/WikiVote.txt"
# 37154, "./data/phy.txt"
# 75888, "./data/Epin.txt"
**/

//using namespace std;

int main(int argn, char **argv)
{
    string file_name = "./data/chep.txt";
    double p = 0.01, alpha = 0.7, lamda = 5.0;
    int R = 10000, MaxK = 50, N = 15233;
    string methods = "cim_celf";
    vector<int> ans;
    int convert_data[] = {1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    for(int i = 0; i < argn; ++i){
        if(argv[i] == string("-help") || argv[i] == string("--help") || argn == 1){
            cout<<"./*** -n ** -datafile ** -k ** -p ** -r ** -m *** -a *** -lamda ***"<<endl;
            return 0;
        }
        if(argv[i] == string("-n"))
            N = atoi(argv[i + 1]);
        if(argv[i] == string("-datafile"))
            file_name = argv[i + 1];
        if(argv[i] == string("-k"))
            MaxK = atoi(argv[i + 1]);
        if(argv[i] == string("-p"))
            p = atof(argv[i + 1]);
        if(argv[i] == string("-r"))
            R = atoi(argv[i + 1]);
        if(argv[i] == string("-m"))
            methods = argv[i + 1];
        if(argv[i] == string("-a"))
            alpha = atof(argv[i + 1]);
        if(argv[i] == string("-lamda"))
            lamda = atof(argv[i + 1]);
    }
    srand((unsigned)time(NULL));
    Graph gph(N);
    Solution solution;
    clock_t tstart, tstop;
    gph.read_graph(file_name, p);
	gph.set_weight();
    tstart = clock();
    if(methods == "greedy_wc")
        ans = solution.greedy_wc(gph, p, R, MaxK);
    else if(methods == "celf_wc")
        ans = solution.celf_wc(gph, p, R, MaxK);
    /*else if(methods == "irie")
        ans = solution.irie(gph, MaxK, p, alpha);
    else if(methods == "page_rank")
        ans = solution.page_rank(gph, MaxK, 0.85);
    else if(methods == "random")
        ans = solution.random(gph, MaxK);
    else if(methods == "single_discount")
        ans = solution.single_discount(gph, MaxK);
    else if(methods == "degree_discount")
        ans = solution.degree_discount(gph, MaxK, p);
    else if(methods == "density_peak")
        ans = solution.density_peak(gph, MaxK);*/
    else if(methods == "degree_out")
        ans = solution.degree_out(gph, MaxK);
    else if(methods == "degree")
        ans = solution.degree(gph, MaxK);

    else if(methods == "cim_greedy")
        ans = solution.cim_greedy(gph, p, R, MaxK, lamda);
    else if(methods == "cim_celf")
        ans = solution.cim_celf(gph, p, R, MaxK, lamda);

    else{
        cout<<"Please input the correct method name"<<endl;
        return 0;
    }
    //solution.write_sps_to_file("./path.txt");
    tstop = clock();
    //cout<<"#influence spread under ic model = "<<solution.monte_carlo_diffusions_ic(gph, ans, p, R)<<endl;
    cout<<MaxK<<','<<(double)(tstop - tstart) / CLOCKS_PER_SEC<<"s"<<endl;
    return 0;
}
