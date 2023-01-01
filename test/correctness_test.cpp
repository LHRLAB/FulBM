#include <gtest/gtest.h>
#include "../src/bdg.h"
#include <random>
#include <tuple>
#include <set>

namespace bdg {
const char *filename = "../data/graph_example1.tsv";
int num_landmarks = 5;

class COR_TEST : public ::testing::Test {
protected:
    void SetUp() override {
        bdg_exp = new BDGraph(filename, false);
        bdg_exp -> InitLabelling(num_landmarks);

        graph_matrix.resize(bdg_exp -> num_v_, std::vector<bool> (bdg_exp -> num_v_, false));
        for (int x = 0; x < bdg_exp -> num_v_; x++) {
            for (auto y: bdg_exp -> adj_list_[x]) {
                graph_matrix[x][y] = true;
                graph_matrix[y][x] = true;
            }
        }
    }

    std::vector<std::vector<uint8_t>> GetLabel() {
        std::vector<std::vector<uint8_t>> res (bdg_exp -> num_v_, std::vector<uint8_t> (bdg_exp -> num_landmarks_));
        for (int i = 0; i < bdg_exp -> num_v_; ++i) {
            for (int j = 0; j < bdg_exp -> num_landmarks_; ++j) {
                res[i][j] = bdg_exp -> labels_[i][j];
            }
        }
        return res;
    }

    std::vector<std::vector<uint8_t>> GetMeta() {
        std::vector<std::vector<uint8_t>> res (bdg_exp -> num_landmarks_, std::vector<uint8_t> (bdg_exp -> num_landmarks_));
        for (int i = 0; i < bdg_exp -> num_landmarks_; ++i) {
            for (int j = 0; j < bdg_exp -> num_landmarks_; ++j) {
                res[i][j] = bdg_exp -> mg_edge_matrix_[i][j];
            }
        }
        return res;
    }

    static void label_test(
            std::vector<std::vector<uint8_t> >& a,
            std::vector<std::vector<uint8_t> >& b) {
        ASSERT_EQ(a.size(), b.size());
        ASSERT_GT(a.size(), 0);
        ASSERT_EQ(a[0].size(), b[0].size());

        for (int i = 0; i < a.size(); i++) {
            for (int j = 0; j < a[0].size(); j++) {
                ASSERT_EQ(a[i][j], b[i][j]) << "labels_[" << i << "][" << j << "]" << " is not equal";
            }
        }
    }

    static void mg_edge_matrix_test(
            std::vector<std::vector<uint8_t> >& a,
            std::vector<std::vector<uint8_t> >& b) {
        ASSERT_EQ(a.size(), b.size());
        ASSERT_GT(a.size(), 0);
        ASSERT_EQ(a[0].size(), b[0].size());

        int num_l = a.size();
        for (int i = 0; i < num_l; i++) {
            for (int j = 0; j < num_l; j++) {
                ASSERT_EQ(a[i][j], b[i][j]) << "mg_edge_matrix_[" << i << "][" << j << "]" << " is not equal";
            }
        }
    }

    void inc_one_edge_test(int start, int end) {
        bdg_exp -> AddEdgeAdjList(start, end);
        bdg_exp -> IncEdgeHandler(start, end);
        auto labels_a = GetLabel();
        auto mg_edge_matrix_a = GetMeta();

        bdg_exp -> Labelling();
        auto labels_b = GetLabel();
        auto mg_edge_matrix_b = GetMeta();

        label_test(labels_a, labels_b);
        mg_edge_matrix_test(mg_edge_matrix_a, mg_edge_matrix_b);
    }

    void dec_one_edge_test(int start, int end) {
        bdg_exp -> DelEdgeAdjList(start, end);
        bdg_exp -> DecEdgeHandler(start, end);
        auto labels_a = GetLabel();
        auto mg_edge_matrix_a = GetMeta();

        bdg_exp -> Labelling();
        auto labels_b = GetLabel();
        auto mg_edge_matrix_b = GetMeta();

        label_test(labels_a, labels_b);
        mg_edge_matrix_test(mg_edge_matrix_a, mg_edge_matrix_b);
    }

    void add_batch_edge_test(std::vector<std::pair<int, int> >& edges) {
        bdg_exp -> AddBatchToGraph(edges);
        bdg_exp -> AddBatchHandler(edges);
        auto labels_a = GetLabel();
        auto mg_edge_matrix_a = GetMeta();

        bdg_exp -> Labelling();
        auto labels_b = GetLabel();
        auto mg_edge_matrix_b = GetMeta();

        label_test(labels_a, labels_b);
        mg_edge_matrix_test(mg_edge_matrix_a, mg_edge_matrix_b);
    }

    void del_batch_edge_test(std::vector<std::pair<int, int> >& edges) {
        bdg_exp -> DelBatchToGraph(edges);
        bdg_exp -> DelBatchHandler(edges);
        auto labels_a = GetLabel();
        auto mg_edge_matrix_a = GetMeta();

        bdg_exp -> Labelling();
        auto labels_b = GetLabel();
        auto mg_edge_matrix_b = GetMeta();

        label_test(labels_a, labels_b);
        mg_edge_matrix_test(mg_edge_matrix_a, mg_edge_matrix_b);
    }

    BDGraph *bdg_exp{};
    std::vector<std::vector<bool>> graph_matrix;
};

TEST_F(COR_TEST, SingleTest) {
    std::default_random_engine e;
    std::uniform_int_distribution<int> d(1, (bdg_exp -> num_v_ - 1));

    for (int t = 0; t < 100000; t++) {
        int x = d(e);
        int y = d(e);
        if (x == y) continue;

        if (graph_matrix[x][y]) {
            dec_one_edge_test(x, y);
        } else {
            inc_one_edge_test(x, y);
        }
        graph_matrix[x][y] = !graph_matrix[x][y];
        graph_matrix[y][x] = !graph_matrix[y][x];
    }
}

TEST_F(COR_TEST, ADBatchTest) {
    std::default_random_engine e;
    std::uniform_int_distribution<int> d(1, bdg_exp -> num_v_ - 1);
    std::uniform_int_distribution<int> d_b(2, graph_matrix.size() - 1);
    for (int t = 0; t < 100000; t++) {
        std::vector<std::pair<int, int>> A, B;
        std::set<std::pair<int, int>> visited;
        int batch_size = d_b(e);
        for (int i = 0; i < batch_size; ++i) {
            int x = d(e);
            int y = d(e);
            if (x == y || visited.count({x, y}) || visited.count({y, x}))
                continue;

            if (graph_matrix[x][y]) {
                A.emplace_back(x, y);
            } else {
                B.emplace_back(x, y);
            }
            visited.emplace(x, y);
            graph_matrix[x][y] = !graph_matrix[x][y];
            graph_matrix[y][x] = !graph_matrix[y][x];
        }
        if (!A.empty()) {
            std::cout << "\rDelete " << A.size() << " edges                   ";
            std::cout.flush();
            del_batch_edge_test(A);
        }
        if (!B.empty()) {
            std::cout << "\rAdd    " << B.size() << " edges                   ";
            std::cout.flush();
            add_batch_edge_test(B);
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

} // namespace bdg