#include <sys/time.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <queue>
#include <cstdint>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <random>
#include <algorithm>
#include <omp.h>

namespace bdg {

class SW_QUEUE {
public:
    inline bool empty();
    inline void push(uint8_t dis_v, int v, bool strong_v);
    inline std::tuple<uint8_t, int, bool> pop();
private:
    std::queue<std::pair<uint8_t, int>> sq, wq;
};

class SW_PRI_QUEUE {
public:
    inline bool empty();
    inline void push(uint8_t dis_v, int v, bool strong_v);
    inline std::tuple<uint8_t, int, bool> pop();
private:
    std::priority_queue<
        std::pair<uint8_t, int>,
        std::vector<std::pair<uint8_t, int>>,
        std::greater<> > pq;
};

class BDGraph {
public:
    BDGraph(const char* filename, bool need_sample);
    ~BDGraph();

    // preprocess
    void AllocateLabel(int num_landmarks);
    void DeallocateLabel();

    clock_t InitLabelling(int num_landmarks);
    void Labelling();

    // check if a vertex is a landmark
    inline bool IsLandMark(int v);

    // updata labels_ and mg_edge_matrix_
    inline void UpdateLabel(int landmark, int v, uint8_t pi, bool strong_v);

    // get distance from a landmark r to vertex v
    inline std::pair<uint8_t, bool> GetDistanceWithStrong(int r, int v);
    static inline uint8_t EncodedDistance(uint8_t dis, bool strong);

    // determine the next affected vertex
    inline bool IsNewAffected(uint8_t dis_v, uint8_t dis_nbr, bool sw_v, bool sw_nbr, int v);
    inline bool IsLAforBDM(int v, uint8_t dis_v, bool strong_v, int r, std::unordered_map<int, uint8_t>& va);

    // get distance of all landmarks from a landmark src
    inline void Dijsktra(int src);
    // update adj_list, assume the change is valid
    void AddEdgeAdjList(int a, int b);
    void DelEdgeAdjList(int a, int b);
    void AddBatchToGraph(std::vector<std::pair<int, int>>& edges);
    void DelBatchToGraph(std::vector<std::pair<int, int>>& edges);

    // handle one edge change, assume the change is valid
    void IncEdgeHandler(int a, int b);
    void DecEdgeHandler(int a, int b);
    // for statistic Va and fan out times
    std::pair<int, int> IEHforVA(int a, int b);
    std::pair<int, int> DEHforVA(int a, int b);

    // handle a batch of edge changes, assume the change is valid
    void AddBatchHandler(std::vector<std::pair<int, int>>& add_edges);
    void DelBatchHandler(std::vector<std::pair<int, int>>& del_edges);
    // for statistic Va and fan out times
    std::pair<int, int> ABHforVA(std::vector<std::pair<int, int>>& add_edges);
    std::pair<int, int> DBHforVA(std::vector<std::pair<int, int>>& del_edges);

    // hybrid batch
    void HyBatchHandler(std::vector<std::pair<int, int>>& add_edges, std::vector<std::pair<int, int>>& del_edges);

    // statistic
    long LabelSize() const;

// Data Structure:
    const uint8_t INF_DISTANCE;   // for 8-bit distances
    const uint8_t WEAK_DISTANCE;   // for 8-bit distances

    int num_v_{};
    int num_landmarks_{};

    std::vector<std::vector<int> > adj_list_;

    std::vector<std::pair<int, int>> eps_;  // for sample

    uint8_t** labels_{};
    uint8_t** mg_edge_matrix_{};

    // the array of landmarks' v_id
    std::vector<int> landmark_set_;
    // map the v_id to the index in landmark_set_
    std::unordered_map<int, int> landmark_map_;

    std::vector<uint8_t> dist_;

    uint8_t **distances{};
    uint8_t **highway{};
};

} // namespace bdg