#include "bdg.h"

namespace bdg {
    bool SW_QUEUE::empty() {
        return sq.empty() && wq.empty();
    }

    void SW_QUEUE::push(uint8_t dis_v, int v, bool strong_v) {
        if (strong_v) {
            sq.emplace(dis_v, v);
        } else {
            wq.emplace(dis_v, v);
        }
    }

    std::tuple<uint8_t, int, bool> SW_QUEUE::pop() {
        if (wq.empty()) {
            auto [dis_v, v] = sq.front(); sq.pop();
            return {dis_v, v, true};
        }
        if (sq.empty()) {
            auto [dis_v, v] = wq.front(); wq.pop();
            return {dis_v, v, false};
        }
        if (sq.front().first <= wq.front().first) {
            auto [dis_v, v] = sq.front(); sq.pop();
            return {dis_v, v, true};
        } else {
            auto [dis_v, v] = wq.front(); wq.pop();
            return {dis_v, v, false};
        }
    }

    inline bool SW_PRI_QUEUE::empty() {
        return pq.empty();
    }

    inline void SW_PRI_QUEUE::push(uint8_t dis_v, int v, bool strong_v) {
        pq.emplace((dis_v << 1) + !strong_v, v);
    }

    inline std::tuple<uint8_t, int, bool> SW_PRI_QUEUE::pop() {
        auto [dis_v, v] = pq.top(); pq.pop();
        return {dis_v >> 1, v, dis_v % 2 == 0};
    }

    BDGraph::BDGraph(const char* filename, bool need_sample) : INF_DISTANCE(125), WEAK_DISTANCE(125) {
        std::ifstream ifs(filename);
        if (!ifs.is_open()) {
            std::cerr << "Cannot open file" << filename << std::endl;
        }

        // get edge set from file
        std::vector<std::pair<int, int> > es;
        std::unordered_map<int, int> vertex2id;
        int v, w;
        num_v_ = 1;
        while (ifs >> v >> w) {
            if (vertex2id.count(v) == 0) { vertex2id[v] = num_v_++;}
            if (vertex2id.count(w) == 0) { vertex2id[w] = num_v_++;}
            v = vertex2id[v];
            w = vertex2id[w];
            es.emplace_back(v, w);
        }
        if (ifs.bad()) {
            std::cerr << "Error reading file" << std::endl;
        }

        // build adjacency list from edge set
        std::vector<std::vector<int>> adj(num_v_);
        for (auto &e : es) {
            adj[e.first].emplace_back(e.second);
            adj[e.second].emplace_back(e.first);
        }

        // sample edge pairs to eps_
        if (need_sample) {
            std::default_random_engine e;
            std::uniform_int_distribution<int> d(1, es.size() - 1);
            std::unordered_set<int> visited;
            for (int i = 0; i < 100001; ) {
                int offset = d(e);
                if (visited.find(offset) != visited.end()) {
                    continue;
                }
                i++;
                eps_.push_back(es[offset]);
                visited.insert(offset);
            }
        }
        std::cout << filename << " V: " << num_v_ << ", E: " << es.size() << std::endl;
        adj_list_ = adj;
    }

    BDGraph::~BDGraph() {
        DeallocateLabel();
    }

    void BDGraph::AllocateLabel(int num_landmarks) {
        num_landmarks_ = num_landmarks;

        labels_ = new uint8_t*[num_v_];
        for (int i = 0; i < num_v_; i++) {
            labels_[i] = new uint8_t[num_landmarks_];
            for (int j = 0; j < num_landmarks_; j++) {
                labels_[i][j] = INF_DISTANCE;
            }
        }

        mg_edge_matrix_ = new uint8_t*[num_landmarks_];
        for (int i = 0; i < num_landmarks_; i++) {
            mg_edge_matrix_[i] = new uint8_t[num_landmarks_];
            for (int j = 0; j < num_landmarks_; j++) {
                mg_edge_matrix_[i][j] = INF_DISTANCE;
            }
        }
    }

    void BDGraph::DeallocateLabel() {
        if (nullptr != labels_ && nullptr != mg_edge_matrix_) {
            for (int i = 0; i < num_v_; i++) {
                delete[] labels_[i];
            }
            delete[] labels_;
            for (int i = 0; i < num_landmarks_; i++) {
                delete[] mg_edge_matrix_[i];
            }
            delete[] mg_edge_matrix_;
        }
        labels_ = nullptr;
        mg_edge_matrix_ = nullptr;
    }

    clock_t BDGraph::InitLabelling(int num_landmarks) {
        DeallocateLabel();
        AllocateLabel(num_landmarks);

        // SelectLandMark
        landmark_map_.clear();
        landmark_set_.clear();
        num_landmarks_ = num_landmarks;
        dist_.clear();
        dist_.resize(num_landmarks_);
        // use the default landmark_choosing strategy, get top NumLandMarks landmarks
        std::vector<std::pair<int, int>> degree(num_v_);
        for (int v = 0; v < num_v_; ++v) {
            degree[v] = std::make_pair(adj_list_[v].size(), v);
        }
        // sort by degree in decreasing order on degree[i].first
        std::nth_element(degree.begin(), degree.begin() + num_landmarks_, degree.end(), [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
            return a.first > b.first;
        });
        for (int i = 0; i < num_landmarks_; ++i) {
            landmark_set_.emplace_back(degree[i].second);
        }
        for (int l = 0; l < num_landmarks_; ++l) {
            landmark_map_[landmark_set_[l]] = l;
        }

        auto start = clock();
        Labelling();
        return clock() - start;
    }

    void BDGraph::Labelling() {
        if (0 == num_landmarks_) return;

        // construct the labelling scheme by BFS, iterate over all landmarks, we can parallelize this loop
        for (int l = 0; l < num_landmarks_; ++l) {
            for (int i = 0; i < num_v_; ++i) {
                labels_[i][l] = INF_DISTANCE;
            }
            for (int i = 0; i < num_landmarks_; ++i) {
                mg_edge_matrix_[i][l] = INF_DISTANCE;
            }
            labels_[landmark_set_[l]][l] = 0;
            mg_edge_matrix_[l][l] = 0;
        }

        for (int l = 0; l < num_landmarks_; ++l) {
            std::queue<int> qNeedLabel, qNotNeedLabel;
            qNeedLabel.push(landmark_set_[l]);

            // the distance from the landmark to all vertex, except the landmark itself, are set to INF_DISTANCE
            std::vector<uint8_t> depth(num_v_, INF_DISTANCE);
            depth[landmark_set_[l]] = 0;
            std::vector<bool> visited(num_v_, false);
            visited[landmark_set_[l]] = true;

            for (uint8_t distance = 1; !qNeedLabel.empty() || !qNotNeedLabel.empty(); ++distance) {
                // in order to compute only the vertices of n_th layer of BFS, store the number of vertices expanded in the previous layer
                unsigned long qNeedLabelSize = qNeedLabel.size();
                unsigned long qNotNeedLabelSize = qNotNeedLabel.size();

                // label the vertices in qNeedLabel in depth n
                for (int i = 0; i < qNeedLabelSize; ++i) {
                    int u = qNeedLabel.front(); qNeedLabel.pop();

                    for (auto &nbr : adj_list_[u]) {
                        if (!visited[nbr]) {
                            visited[nbr] = true;
                            depth[nbr] = distance;
                            if (IsLandMark(nbr)) {
                                qNotNeedLabel.push(nbr);
                                mg_edge_matrix_[landmark_map_[nbr]][l] = depth[nbr];
                            } else {
                                qNeedLabel.push(nbr);
                                labels_[nbr][l] = depth[nbr];
                            }
                        }
                    }
                }
                // label the vertices in qNotNeedLabel in depth n
                for (int i = 0; i < qNotNeedLabelSize; ++i) {
                    int u = qNotNeedLabel.front(); qNotNeedLabel.pop();
                    for (auto &nbr : adj_list_[u]) {
                        if (!visited[nbr]) {
                            visited[nbr] = true;
                            depth[nbr] = distance;
                            qNotNeedLabel.push(nbr);
                            if (IsLandMark(nbr)) {
                                mg_edge_matrix_[landmark_map_[nbr]][l] = WEAK_DISTANCE;
                            } else {
                                labels_[nbr][l] = WEAK_DISTANCE;
                            }
                        }
                    }
                }
            }
        }
    }

    long BDGraph::LabelSize() const {
        long size = 0;
        for (int i = 0; i < num_v_; i++) {
            for (int j = 0; j < num_landmarks_; j++) {
                if (labels_[i][j] != WEAK_DISTANCE) {
                    size ++;
                }
            }
        }
        for (int i = 0; i < num_landmarks_; i++) {
            for (int j = 0; j < num_landmarks_; j++) {
                if (mg_edge_matrix_[i][j] != WEAK_DISTANCE) {
                    size ++;
                }
            }
        }
        return size / (1024 * 1024);
    }

    inline bool BDGraph::IsLandMark(int v) {
        // return std::find(landmark_set_.begin(), landmark_set_.end(), v) != landmark_set_.end();
        return landmark_map_.find(v) != landmark_map_.end();
    }

    inline void BDGraph::UpdateLabel(int landmark, int v, uint8_t dis_v, bool strong_v) {
        if (strong_v) {
            if (IsLandMark(v)) {
                mg_edge_matrix_[landmark_map_[v]][landmark] = dis_v;
            } else {
                labels_[v][landmark] = dis_v;
            }
        } else {
            if (IsLandMark(v)) {
                mg_edge_matrix_[landmark_map_[v]][landmark] = WEAK_DISTANCE;
            } else {
                labels_[v][landmark] = WEAK_DISTANCE;
            }
        }
    }

    inline std::pair<uint8_t, bool> BDGraph::GetDistanceWithStrong(int landmark, int v) {
        if (IsLandMark(v) && mg_edge_matrix_[landmark_map_[v]][landmark] != WEAK_DISTANCE) {
            return std::make_pair(mg_edge_matrix_[landmark_map_[v]][landmark], true);
        }
        if (labels_[v][landmark] != WEAK_DISTANCE) {
            return std::make_pair(labels_[v][landmark], true);
        }

        if (IsLandMark(v)) {
            return std::make_pair(dist_[landmark_map_[v]], false);
        }

        uint8_t pi = INF_DISTANCE;
        for (int r = 0; r < num_landmarks_; ++r) {
            if (labels_[v][r] != WEAK_DISTANCE && dist_[r] != INF_DISTANCE) {
                pi = pi < labels_[v][r] + dist_[r] ? pi : labels_[v][r] + dist_[r];
            }
        }
        return std::make_pair(pi, false);
    }

    inline uint8_t BDGraph::EncodedDistance(uint8_t dis, bool strong) {
        return (dis << 1) + (!strong);
    }

    inline void BDGraph::Dijsktra(int src) {
        std::fill(dist_.begin(), dist_.end(), INF_DISTANCE);
        dist_[src] = 0;

        std::vector<bool> finished(num_landmarks_, false);
        for (int l = 0; l < num_landmarks_; ++l) {
            uint8_t min = INF_DISTANCE;
            int min_index;
            for (int r = 0; r < num_landmarks_; ++r) {
                if (!finished[r] && dist_[r] <= min) {
                    min = dist_[r];
                    min_index = r;
                }
            }
            finished[min_index] = true;

            for (int r = 0; r < num_landmarks_; ++r) {
                if (!finished[r] && mg_edge_matrix_[min_index][r] != WEAK_DISTANCE && dist_[min_index] != INF_DISTANCE) {
                    if (dist_[r] > dist_[min_index] + mg_edge_matrix_[min_index][r]) {
                        dist_[r] = dist_[min_index] + mg_edge_matrix_[min_index][r];
                    }
                }
            }
        }
    }

    void BDGraph::AddEdgeAdjList(int a, int b) {
        adj_list_[a].emplace_back(b);
        adj_list_[b].emplace_back(a);
    }

    void BDGraph::DelEdgeAdjList(int a, int b) {
        for (size_t index = 0; index < adj_list_[a].size() - 1; index++) {
            if(adj_list_[a][index] == b) {
                std::swap(adj_list_[a][index], adj_list_[a][adj_list_[a].size() - 1]);
                break;
            }
        }
        for (size_t index = 0; index < adj_list_[b].size() - 1; index++) {
            if(adj_list_[b][index] == a) {
                std::swap(adj_list_[b][index], adj_list_[b][adj_list_[b].size() - 1]);
                break;
            }
        }
        adj_list_[a].pop_back();
        adj_list_[b].pop_back();
    }

    inline bool BDGraph::IsNewAffected(uint8_t dis_v, uint8_t dis_nbr, bool sw_v, bool sw_nbr, int v) {
        return dis_v + 1 < dis_nbr || (dis_v + 1 == dis_nbr && sw_v && !sw_nbr && !IsLandMark(v));
    }

    void BDGraph::IncEdgeHandler(int a, int b) {
        for (int landmark = 0; landmark < num_landmarks_; ++landmark) {
            SW_QUEUE q;
            std::unordered_set<int> visited;
            Dijsktra(landmark);

            auto [dis_a, strong_a] = GetDistanceWithStrong(landmark, a);
            auto [dis_b, strong_b] = GetDistanceWithStrong(landmark, b);
            if (IsNewAffected(dis_a, dis_b, strong_a, strong_b, a) || a == landmark_set_[landmark]) {
                q.push(dis_a + 1, b, (!IsLandMark(a) || a == landmark_set_[landmark]) && strong_a);
            } else if (IsNewAffected(dis_b, dis_a, strong_b, strong_a, b) || b == landmark_set_[landmark]) {
                q.push(dis_b + 1, a, (!IsLandMark(b) || b == landmark_set_[landmark]) && strong_b);
            } else {
                continue;
            }

            while (!q.empty()) {
                auto [dis_v, v, strong_v] = q.pop();

                if (visited.find(v) != visited.end()) continue;
                UpdateLabel(landmark, v, dis_v, strong_v);
                visited.insert(v);

                for (auto &nbr : adj_list_[v]) {
                    if (visited.find(nbr) == visited.end()) {
                        auto [dis_nbr, strong_nbr] = GetDistanceWithStrong(landmark, nbr);
                        if (IsNewAffected(dis_v, dis_nbr, strong_v, strong_nbr, v)) {
                            q.push(dis_v + 1, nbr, !IsLandMark(v) && strong_v);
                        }
                    }
                }
            }
        }
    }

    void BDGraph::DecEdgeHandler(int a, int b) {
        // update labels of each landmark
        std::vector<std::unordered_map<int, uint8_t>> upsert_labels(num_landmarks_);
        std::vector<std::unordered_map<int, bool>> is_strongs(num_landmarks_);
        for (int landmark = 0; landmark < num_landmarks_; landmark++) {
            std::unordered_map<int, uint8_t> affected;
            std::priority_queue<
                    std::pair<uint8_t, int>,
                    std::vector<std::pair<uint8_t, int>>,
                    std::greater<>
            >pq;
            std::queue<std::tuple<uint8_t, int, bool>> q;

            Dijsktra(landmark);

            // find anchor vertices
            auto [dis_a, strong_a] = GetDistanceWithStrong(landmark, a);
            auto [dis_b, strong_b] = GetDistanceWithStrong(landmark, b);
            if (dis_a > dis_b && IsLAforBDM(a, dis_a, strong_a, landmark, affected)) {
                q.emplace(dis_a, a, strong_a);
                affected[a] = INF_DISTANCE;
            } else if (dis_a < dis_b && IsLAforBDM(b, dis_b, strong_b, landmark, affected)) {
                q.emplace(dis_b, b, strong_b);
                affected[b] = INF_DISTANCE;
            } else {
                continue;
            }

            // find affected vertices
            while (!q.empty()) {
                auto [dis_v, v, strong_v] = q.front(); q.pop();

                uint8_t min_pi = INF_DISTANCE;
                bool strong = false;
                for (auto &nbr : adj_list_[v]) {
                    if (affected.find(nbr) == affected.end()) {
                        auto [dis_nbr, strong_nbr] = GetDistanceWithStrong(landmark, nbr);
                        if (dis_nbr > dis_v && IsLAforBDM(nbr, dis_nbr, strong_nbr, landmark, affected)) {
                            q.emplace(dis_nbr, nbr, strong_nbr);
                            affected[nbr] = INF_DISTANCE;
                        } else {
                            if (dis_nbr + 1 < min_pi) {
                                min_pi = dis_nbr + 1;
                                strong = (!IsLandMark(nbr)) && strong_nbr;
                            } else if (dis_nbr + 1 == min_pi ) {
                                strong |= strong_nbr && !IsLandMark(nbr);
                            }
                        }
                    }
                }
                upsert_labels[landmark][v] = min_pi;
                is_strongs[landmark][v] = strong;

                if (min_pi != INF_DISTANCE) {
                    pq.emplace(min_pi, v);
                    affected[v] = min_pi;
                }
            }

            // record new labels
            while (!pq.empty()) {
                auto [dis_v, v] = pq.top(); pq.pop();
                if (affected.find(v) == affected.end()) {
                    continue;
                }
                for (auto &nbr : adj_list_[v]) {
                    if (affected.find(nbr) != affected.end()) {
                        if (affected[nbr] > dis_v + 1 || (affected[nbr] == dis_v + 1 && !is_strongs[landmark][nbr] && !IsLandMark(v) && is_strongs[landmark][v])) {
                            is_strongs[landmark][nbr] = (!IsLandMark(v)) && is_strongs[landmark][v];
                            affected[nbr] = dis_v + 1;
                            pq.emplace(dis_v + 1, nbr);
                        }
                    }
                }
                upsert_labels[landmark][v] = dis_v;
                affected.erase(v);
            }
        }
        // update labels
        for (int landmark = 0; landmark < num_landmarks_; ++landmark) {
            for (auto [v, pi] : upsert_labels[landmark]) {
                if (IsLandMark(v)) {
                    if (is_strongs[landmark][v]) {
                        mg_edge_matrix_[landmark_map_[v]][landmark] = pi;
                    } else {
                        mg_edge_matrix_[landmark_map_[v]][landmark] = WEAK_DISTANCE;
                    }
                } else {
                    if (is_strongs[landmark][v]) {
                        labels_[v][landmark] = pi;
                    } else {
                        labels_[v][landmark] = WEAK_DISTANCE;
                    }
                }
            }
        }
    }

    std::pair<int, int> BDGraph::IEHforVA(int a, int b) {
        int va = 0, fo = 0;
        for (int landmark = 0; landmark < num_landmarks_; ++landmark) {
            SW_QUEUE q;
            std::unordered_set<int> visited;
            Dijsktra(landmark);

            auto [dis_a, strong_a] = GetDistanceWithStrong(landmark, a);
            auto [dis_b, strong_b] = GetDistanceWithStrong(landmark, b);
            if (IsNewAffected(dis_a, dis_b, strong_a, strong_b, a) || a == landmark_set_[landmark]) {
                q.push(dis_a + 1, b, (!IsLandMark(a) || a == landmark_set_[landmark]) && strong_a);
            } else if (IsNewAffected(dis_b, dis_a, strong_b, strong_a, b) || b == landmark_set_[landmark]) {
                q.push(dis_b + 1, a, (!IsLandMark(b) || b == landmark_set_[landmark]) && strong_b);
            } else {
                continue;
            }

            while (!q.empty()) {
                auto [dis_v, v, strong_v] = q.pop();

                if (visited.find(v) != visited.end()) continue;
                va++;
                UpdateLabel(landmark, v, dis_v, strong_v);
                visited.insert(v);

                for (auto &nbr : adj_list_[v]) {
                    if (visited.find(nbr) == visited.end()) {
                        auto [dis_nbr, strong_nbr] = GetDistanceWithStrong(landmark, nbr);
                        if (IsNewAffected(dis_v, dis_nbr, strong_v, strong_nbr, v)) {
                            q.push(dis_v + 1, nbr, !IsLandMark(v) && strong_v);
                        }
                    }
                }
            }
        }
        return {va, fo};
    }

    std::pair<int, int> BDGraph::DEHforVA(int a, int b) {
        int va = 0, fo = 0;
        std::vector<std::unordered_map<int, uint8_t>> upsert_labels(num_landmarks_);
        std::vector<std::unordered_map<int, bool>> is_strongs(num_landmarks_);
        for (int landmark = 0; landmark < num_landmarks_; landmark++) {
            std::unordered_map<int, uint8_t> affected;
            std::priority_queue<
                    std::pair<uint8_t, int>,
                    std::vector<std::pair<uint8_t, int>>,
                    std::greater<>
            >pq;
            std::queue<std::tuple<uint8_t, int, bool>> q;

            Dijsktra(landmark);

            // find anchor vertices
            auto [dis_a, strong_a] = GetDistanceWithStrong(landmark, a);
            auto [dis_b, strong_b] = GetDistanceWithStrong(landmark, b);
            fo++;
            if (dis_a > dis_b && IsLAforBDM(a, dis_a, strong_a, landmark, affected)) {
                q.emplace(dis_a, a, strong_a);
                affected[a] = INF_DISTANCE;
            } else if (dis_a < dis_b && IsLAforBDM(b, dis_b, strong_b, landmark, affected)) {
                q.emplace(dis_b, b, strong_b);
                affected[b] = INF_DISTANCE;
            } else {
                continue;
            }

            // find affected vertices
            while (!q.empty()) {
                auto [dis_v, v, strong_v] = q.front(); q.pop();

                uint8_t min_pi = INF_DISTANCE;
                bool strong = false;
                va ++; fo++;
                for (auto &nbr : adj_list_[v]) {
                    if (affected.find(nbr) == affected.end()) {
                        auto [dis_nbr, strong_nbr] = GetDistanceWithStrong(landmark, nbr);
                        fo++;
                        if (dis_nbr > dis_v && IsLAforBDM(nbr, dis_nbr, strong_nbr, landmark, affected)) {
                            q.emplace(dis_nbr, nbr, strong_nbr);
                            affected[nbr] = INF_DISTANCE;
                        } else {
                            if (dis_nbr + 1 < min_pi) {
                                min_pi = dis_nbr + 1;
                                strong = (!IsLandMark(nbr)) && strong_nbr;
                            } else if (dis_nbr + 1 == min_pi ) {
                                strong |= strong_nbr && !IsLandMark(nbr);
                            }
                        }
                    }
                }
                upsert_labels[landmark][v] = min_pi;
                is_strongs[landmark][v] = strong;

                if (min_pi != INF_DISTANCE) {
                    pq.emplace(min_pi, v);
                    affected[v] = min_pi;
                }
            }

            // record new labels
            while (!pq.empty()) {
                auto [dis_v, v] = pq.top(); pq.pop();
                if (affected.find(v) == affected.end()) {
                    continue;
                }
                fo++;
                for (auto &nbr : adj_list_[v]) {
                    if (affected.find(nbr) != affected.end()) {
                        if (affected[nbr] > dis_v + 1 || (affected[nbr] == dis_v + 1 && !is_strongs[landmark][nbr] && !IsLandMark(v) && is_strongs[landmark][v])) {
                            is_strongs[landmark][nbr] = (!IsLandMark(v)) && is_strongs[landmark][v];
                            affected[nbr] = dis_v + 1;
                            pq.emplace(dis_v + 1, nbr);
                        }
                    }
                }
                upsert_labels[landmark][v] = dis_v;
                affected.erase(v);
            }
        }
        // update labels
        for (int landmark = 0; landmark < num_landmarks_; ++landmark) {
            for (auto [v, pi] : upsert_labels[landmark]) {
                if (IsLandMark(v)) {
                    if (is_strongs[landmark][v]) {
                        mg_edge_matrix_[landmark_map_[v]][landmark] = pi;
                    } else {
                        mg_edge_matrix_[landmark_map_[v]][landmark] = WEAK_DISTANCE;
                    }
                } else {
                    if (is_strongs[landmark][v]) {
                        labels_[v][landmark] = pi;
                    } else {
                        labels_[v][landmark] = WEAK_DISTANCE;
                    }
                }
            }
        }

        return {va, fo};
    }

    void BDGraph::AddBatchToGraph(std::vector<std::pair<int, int>>& add_edges) {
        for (auto [a, b] : add_edges) {
            AddEdgeAdjList(a, b);
        }
    }

    void BDGraph::AddBatchHandler(std::vector<std::pair<int, int>>& add_edges) {
        std::vector<std::unordered_map<int, uint8_t>> upsert_labels(num_landmarks_);
        for (int landmark = 0; landmark < num_landmarks_; landmark++) {
            // star is the end vertex of added edge that farther from landmark
            SW_PRI_QUEUE pq;
            std::unordered_set<int> visited;
            Dijsktra(landmark);

            for (auto [a, b]: add_edges) {
                auto [dis_a, strong_a] = GetDistanceWithStrong(landmark, a);
                auto [dis_b, strong_b] = GetDistanceWithStrong(landmark, b);
                if (IsNewAffected(dis_a, dis_b, strong_a, strong_b, a) || a == landmark_set_[landmark]) {
                    pq.push(dis_a + 1, b, (!IsLandMark(a) || a == landmark_set_[landmark]) && strong_a);
                } else if (IsNewAffected(dis_b, dis_a, strong_b, strong_a, b) || b == landmark_set_[landmark]) {
                    pq.push(dis_b + 1, a, (!IsLandMark(b) || b == landmark_set_[landmark]) && strong_b);
                } else {
                    continue;
                }
            }

            while (!pq.empty()) {
                auto [dis_v, v, strong_v] = pq.pop();

                if (visited.find(v) != visited.end()) continue;
                upsert_labels[landmark][v] = strong_v ? dis_v : WEAK_DISTANCE;
                visited.insert(v);

                for (auto &nbr : adj_list_[v]) {
                    if (visited.find(nbr) == visited.end()) {
                        auto [dis_nbr, strong_nbr] = GetDistanceWithStrong(landmark, nbr);
                        if (IsNewAffected(dis_v, dis_nbr, strong_v, strong_nbr, v)) {
                            pq.push(dis_v + 1, nbr, !IsLandMark(v) && strong_v);
                        }
                    }
                }
            }
        }
        // update labels
        for (int landmark = 0; landmark < num_landmarks_; ++landmark) {
            for (auto [v, pi] : upsert_labels[landmark]) {
                if (IsLandMark(v)) {
                    mg_edge_matrix_[landmark_map_[v]][landmark] = pi;
                } else {
                    labels_[v][landmark] = pi;
                }
            }
        }
    }

    std::pair<int, int> BDGraph::ABHforVA(std::vector<std::pair<int, int>>& add_edges) {
        int va = 0;
        std::vector<std::unordered_map<int, uint8_t>> upsert_labels(num_landmarks_);
        std::vector<std::unordered_map<int, bool>> is_strongs(num_landmarks_);
        for (int landmark = 0; landmark < num_landmarks_; landmark++) {
            // star is the end vertex of added edge that farther from landmark
            SW_PRI_QUEUE pq;
            std::unordered_set<int> visited;
            Dijsktra(landmark);

            for (auto [a, b]: add_edges) {
                auto [dis_a, strong_a] = GetDistanceWithStrong(landmark, a);
                auto [dis_b, strong_b] = GetDistanceWithStrong(landmark, b);
                if (IsNewAffected(dis_a, dis_b, strong_a, strong_b, a) || a == landmark_set_[landmark]) {
                    pq.push(dis_a + 1, b, (!IsLandMark(a) || a == landmark_set_[landmark]) && strong_a);
                } else if (IsNewAffected(dis_b, dis_a, strong_b, strong_a, b) || b == landmark_set_[landmark]) {
                    pq.push(dis_b + 1, a, (!IsLandMark(b) || b == landmark_set_[landmark]) && strong_b);
                } else {
                    continue;
                }
            }

            while (!pq.empty()) {
                auto [dis_v, v, strong_v] = pq.pop();

                if (visited.find(v) != visited.end()) continue;
                upsert_labels[landmark][v] = dis_v;
                is_strongs[landmark][v] = strong_v;
                va++;
                visited.insert(v);

                for (auto &nbr : adj_list_[v]) {
                    if (visited.find(nbr) == visited.end()) {
                        auto [dis_nbr, strong_nbr] = GetDistanceWithStrong(landmark, nbr);
                        if (IsNewAffected(dis_v, dis_nbr, strong_v, strong_nbr, v)) {
                            pq.push(dis_v + 1, nbr, !IsLandMark(v) && strong_v);
                        }
                    }
                }
            }
        }
        // update labels
        for (int landmark = 0; landmark < num_landmarks_; ++landmark) {
            for (auto [v, pi] : upsert_labels[landmark]) {
                if (IsLandMark(v)) {
                    if (is_strongs[landmark][v]) {
                        mg_edge_matrix_[landmark_map_[v]][landmark] = pi;
                    } else {
                        mg_edge_matrix_[landmark_map_[v]][landmark] = WEAK_DISTANCE;
                    }
                } else {
                    if (is_strongs[landmark][v]) {
                        labels_[v][landmark] = pi;
                    } else {
                        labels_[v][landmark] = WEAK_DISTANCE;
                    }
                }
            }
        }
        return {va, va};
    }

    void BDGraph::DelBatchToGraph(std::vector<std::pair<int, int>>& del_edges) {
        for (auto [a, b] : del_edges) {
            DelEdgeAdjList(a, b);
        }
    }

    void BDGraph::DelBatchHandler(std::vector<std::pair<int, int>>& del_edges) {
        std::vector<std::unordered_map<int, uint8_t>> upsert_labels(num_landmarks_);
        std::vector<std::unordered_map<int, bool>> is_strongs(num_landmarks_);
        for (int landmark = 0; landmark < num_landmarks_; landmark++) {
            std::unordered_map<int, uint8_t> affected;
            std::priority_queue<
                    std::pair<uint8_t, int>,
                    std::vector<std::pair<uint8_t, int>>,
                    std::greater<>
            > pq;
            std::priority_queue<
                    std::tuple<uint8_t, int, bool>,
                    std::vector<std::tuple<uint8_t, int, bool>>,
                    std::greater<>
            > expq;
            Dijsktra(landmark);

            // find anchor vertices
            for (auto [a, b] : del_edges) {
                auto [dis_a, strong_a] = GetDistanceWithStrong(landmark, a);
                auto [dis_b, strong_b] = GetDistanceWithStrong(landmark, b);
                if (dis_a > dis_b && IsLAforBDM(a, dis_a, strong_a, landmark, affected)) {
                    expq.emplace(dis_a, a, strong_a);
                    affected[a] = INF_DISTANCE;
                } else if (dis_a < dis_b && IsLAforBDM(b, dis_b, strong_b, landmark, affected)) {
                    expq.emplace(dis_b, b, strong_b);
                    affected[b] = INF_DISTANCE;
                } else {
                    continue;
                }
            }

            // find affected vertices
            while (!expq.empty()) {
                auto [dis_v, v, strong_v] = expq.top(); expq.pop();

                uint8_t min_pi = INF_DISTANCE;
                bool strong = false;
                for (auto &nbr : adj_list_[v]) {
                    if (affected.find(nbr) == affected.end()) {
                        auto [dis_nbr, strong_nbr] = GetDistanceWithStrong(landmark, nbr);
                        if (dis_nbr > dis_v && IsLAforBDM(nbr, dis_nbr, strong_nbr, landmark, affected)) {
                            expq.emplace(dis_nbr, nbr, strong_nbr);
                            affected[nbr] = INF_DISTANCE;
                        } else {
                            if (dis_nbr + 1 < min_pi) {
                                min_pi = dis_nbr + 1;
                                strong = (!IsLandMark(nbr)) && strong_nbr;
                            } else if (dis_nbr + 1 == min_pi) {
                                strong |= strong_nbr && !IsLandMark(nbr);
                            }
                        }
                    }
                }
                upsert_labels[landmark][v] = min_pi;
                is_strongs[landmark][v] = strong;

                if (min_pi != INF_DISTANCE) {
                    pq.emplace(min_pi, v);
                    affected[v] = min_pi;
                }
            }

            // record new labels
            while (!pq.empty()) {
                auto [dis_v, v] = pq.top(); pq.pop();
                if (affected.find(v) == affected.end()) {
                    continue;
                }
                for (auto &nbr : adj_list_[v]) {
                    if (affected.find(nbr) != affected.end()) {
                        if (affected[nbr] > dis_v + 1 || (affected[nbr] == dis_v + 1 && !is_strongs[landmark][nbr] && !IsLandMark(v) && is_strongs[landmark][v])) {
                            is_strongs[landmark][nbr] = (!IsLandMark(v)) && is_strongs[landmark][v];
                            affected[nbr] = dis_v + 1;
                            pq.emplace(dis_v + 1, nbr);
                        }
                    }
                }
                upsert_labels[landmark][v] = dis_v;
                affected.erase(v);
            }
        }
        // update labels
        for (int landmark = 0; landmark < num_landmarks_; ++landmark) {
            for (auto [v, pi] : upsert_labels[landmark]) {
                if (IsLandMark(v)) {
                    if (is_strongs[landmark][v]) {
                        mg_edge_matrix_[landmark_map_[v]][landmark] = pi;
                    } else {
                        mg_edge_matrix_[landmark_map_[v]][landmark] = WEAK_DISTANCE;
                    }
                } else {
                    if (is_strongs[landmark][v]) {
                        labels_[v][landmark] = pi;
                    } else {
                        labels_[v][landmark] = WEAK_DISTANCE;
                    }
                }
            }
        }
    }

    std::pair<int, int> BDGraph::DBHforVA(std::vector<std::pair<int, int>> &del_edges) {
        std::vector<std::unordered_map<int, uint8_t>> upsert_labels(num_landmarks_);
        std::vector<std::unordered_map<int, bool>> is_strongs(num_landmarks_);
        int va = 0, fo = 0;
        for (int landmark = 0; landmark < num_landmarks_; landmark++) {
            std::unordered_map<int, uint8_t> affected;
            std::priority_queue<
                    std::pair<uint8_t, int>,
                    std::vector<std::pair<uint8_t, int>>,
                    std::greater<>
            > pq;
            std::priority_queue<
                    std::tuple<uint8_t, int, bool>,
                    std::vector<std::tuple<uint8_t, int, bool>>,
                    std::greater<>
            > expq;
            Dijsktra(landmark);

            // find anchor vertices
            for (auto [a, b] : del_edges) {
                auto [dis_a, strong_a] = GetDistanceWithStrong(landmark, a);
                auto [dis_b, strong_b] = GetDistanceWithStrong(landmark, b);
                fo++;
                if (dis_a > dis_b && IsLAforBDM(a, dis_a, strong_a, landmark, affected)) {
                    expq.emplace(dis_a, a, strong_a);
                    affected[a] = INF_DISTANCE;
                } else if (dis_a < dis_b && IsLAforBDM(b, dis_b, strong_b, landmark, affected)) {
                    expq.emplace(dis_b, b, strong_b);
                    affected[b] = INF_DISTANCE;
                } else {
                    continue;
                }
            }

            // find affected vertices
            while (!expq.empty()) {
                auto [dis_v, v, strong_v] = expq.top(); expq.pop();

                uint8_t min_pi = INF_DISTANCE;
                bool strong = false;
                va++, fo++;
                for (auto &nbr : adj_list_[v]) {
                    if (affected.find(nbr) == affected.end()) {
                        auto [dis_nbr, strong_nbr] = GetDistanceWithStrong(landmark, nbr);
                        fo++;
                        if (dis_nbr > dis_v && IsLAforBDM(nbr, dis_nbr, strong_nbr, landmark, affected)) {
                            expq.emplace(dis_nbr, nbr, strong_nbr);
                            affected[nbr] = INF_DISTANCE;
                        } else {
                            if (dis_nbr + 1 < min_pi) {
                                min_pi = dis_nbr + 1;
                                strong = (!IsLandMark(nbr)) && strong_nbr;
                            } else if (dis_nbr + 1 == min_pi) {
                                strong |= strong_nbr && !IsLandMark(nbr);
                            }
                        }
                    }
                }
                upsert_labels[landmark][v] = min_pi;
                is_strongs[landmark][v] = strong;

                if (min_pi != INF_DISTANCE) {
                    pq.emplace(min_pi, v);
                    affected[v] = min_pi;
                }
            }

            // record new labels
            while (!pq.empty()) {
                auto [dis_v, v] = pq.top(); pq.pop();
                if (affected.find(v) == affected.end()) {
                    continue;
                }
                fo++;
                for (auto &nbr : adj_list_[v]) {
                    if (affected.find(nbr) != affected.end()) {
                        if (affected[nbr] > dis_v + 1 ||
                            (affected[nbr] == dis_v + 1 && !is_strongs[landmark][nbr] && !IsLandMark(v) && is_strongs[landmark][v])) {
                            is_strongs[landmark][nbr] = (!IsLandMark(v)) && is_strongs[landmark][v];
                            affected[nbr] = dis_v + 1;
                            pq.emplace(dis_v + 1, nbr);
                        }
                    }
                }
                upsert_labels[landmark][v] = dis_v;
                affected.erase(v);
            }
        }
        // update labels
        for (int landmark = 0; landmark < num_landmarks_; ++landmark) {
            for (auto [v, pi] : upsert_labels[landmark]) {
                if (IsLandMark(v)) {
                    if (is_strongs[landmark][v]) {
                        mg_edge_matrix_[landmark_map_[v]][landmark] = pi;
                    } else {
                        mg_edge_matrix_[landmark_map_[v]][landmark] = WEAK_DISTANCE;
                    }
                } else {
                    if (is_strongs[landmark][v]) {
                        labels_[v][landmark] = pi;
                    } else {
                        labels_[v][landmark] = WEAK_DISTANCE;
                    }
                }
            }
        }
        return {va, fo};
    }

    inline bool BDGraph::IsLAforBDM(int v, uint8_t dis_v, bool strong_v, int r, std::unordered_map<int, uint8_t> &va) {
        bool strong_temp = false;
        for (auto &nbr: adj_list_[v]) {
            if (va.find(nbr) == va.end()) {
                auto [dis_nbr, strong_nbr] = GetDistanceWithStrong(r, nbr);
                if (dis_nbr < dis_v) {
                    strong_temp |= (strong_nbr && !IsLandMark(nbr));
                    if (strong_temp == strong_v) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    void BDGraph::HyBatchHandler(std::vector<std::pair<int, int>>& add_edges,
                                 std::vector<std::pair<int, int>>& del_edges) {
        std::vector<std::unordered_map<int, uint8_t>> upsert_labels(num_landmarks_);
        std::vector<std::unordered_map<int, bool>> is_strongs(num_landmarks_);
        for (int landmark = 0; landmark < num_landmarks_; landmark++) {
            std::vector<uint8_t> affected(num_v_, 255);
            std::priority_queue<
                    std::pair<uint8_t, int>,
                    std::vector<std::pair<uint8_t, int>>,
                    std::greater<>
            > expq, pq;

            Dijsktra(landmark);

            // find anchor vertices
            for (auto [a, b] : add_edges) {
                auto [dis_a, strong_a] = GetDistanceWithStrong(landmark, a);
                auto [dis_b, strong_b] = GetDistanceWithStrong(landmark, b);
                if (EncodedDistance(dis_a + 1, strong_a) < EncodedDistance(dis_b, strong_b)) {
                    expq.emplace(dis_a + 1, b);
                    affected[b] = INF_DISTANCE;
                } else if (EncodedDistance(dis_b + 1, strong_b) < EncodedDistance(dis_a, strong_a)) {
                    expq.emplace(dis_b + 1, a);
                    affected[a] = INF_DISTANCE;
                } else {
                    continue;
                }
            }
            for (auto [a, b] : del_edges) {
                auto [dis_a, strong_a] = GetDistanceWithStrong(landmark, a);
                auto [dis_b, strong_b] = GetDistanceWithStrong(landmark, b);
                if (dis_a > dis_b) {
                    expq.emplace(dis_a, a);
                    affected[a] = INF_DISTANCE;
                } else if (dis_a < dis_b) {
                    expq.emplace(dis_b, b);
                    affected[b] = INF_DISTANCE;
                } else {
                    continue;
                }
            }

            // find affected vertices
            while (!expq.empty()) {
                auto [dis_v, v] = expq.top(); expq.pop();

                uint8_t min_pi = INF_DISTANCE;
                bool strong = false;

                for (auto &nbr : adj_list_[v]) {
                    if (affected[nbr] == 255) {
                        auto [dis_nbr, strong_nbr] = GetDistanceWithStrong(landmark, nbr);
                        if (dis_nbr > dis_v) {
                            expq.emplace(dis_v + 1, nbr);
                            affected[nbr] = INF_DISTANCE;
                        } else {
                            if (dis_nbr + 1 < min_pi) {
                                min_pi = dis_nbr + 1;
                                if (!IsLandMark(nbr)) {
                                    strong = strong_nbr;
                                } else {
                                    strong = (landmark_set_[landmark] == nbr);
                                }
                            } else if (dis_nbr + 1 == min_pi) {
                                strong |= (!IsLandMark(nbr)) && strong_nbr;
                                strong |= (nbr == landmark_set_[landmark]);
                            }
                        }
                    }
                }

                upsert_labels[landmark][v] = min_pi;
                is_strongs[landmark][v] = strong;

                if (min_pi != INF_DISTANCE) {
                    pq.emplace(min_pi, v);
                    affected[v] = min_pi;
                }
            }

            // record new labels
            while (!pq.empty()) {
                auto [dis_v, v] = pq.top(); pq.pop();
                if (affected[v] == 255) {
                    continue;
                }
                for (auto &nbr : adj_list_[v]) {
                    if (affected[nbr] != 255) {
                        if (affected[nbr] == dis_v + 1 && !is_strongs[landmark][nbr] && !IsLandMark(v) && is_strongs[landmark][v]) {
                            pq.emplace(dis_v + 1, nbr);
                            is_strongs[landmark][nbr] = true;
                        }
                        if (affected[nbr] > dis_v + 1) {
                            is_strongs[landmark][nbr] = (!IsLandMark(v)) && is_strongs[landmark][v];
                            affected[nbr] = dis_v + 1;
                            pq.emplace(dis_v + 1, nbr);
                        }
                    }
                }
                upsert_labels[landmark][v] = dis_v;
                affected[v] = 255;
            }
        }

        // update labels
        for (int landmark = 0; landmark < num_landmarks_; ++landmark) {
            for (auto [v, pi] : upsert_labels[landmark]) {
                if (IsLandMark(v)) {
                    if (is_strongs[landmark][v]) {
                        mg_edge_matrix_[landmark_map_[v]][landmark] = pi;
                    } else {
                        mg_edge_matrix_[landmark_map_[v]][landmark] = WEAK_DISTANCE;
                    }
                } else {
                    if (is_strongs[landmark][v]) {
                        labels_[v][landmark] = pi;
                    } else {
                        labels_[v][landmark] = WEAK_DISTANCE;
                    }
                }
            }
        }
    }
} // namespace bdg