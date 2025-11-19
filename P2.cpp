#include "Polymer.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>

struct PolymerStats {
    std::vector<double> R2_P;
    std::vector<double> R2_g;
    std::vector<int> counts;
    std::vector<double> weights;
    
    PolymerStats(int maxN) {
        R2_P.resize(maxN + 1, 0.0);
        R2_g.resize(maxN + 1, 0.0);
        counts.resize(maxN + 1, 0);
        weights.resize(maxN + 1, 0.0);
    }
};

void simulateIdealPolymer(int dimension, int M, int Nmax, const std::string& filename) {
    std::cout << "Simulating Ideal Polymer (dim=" << dimension << ", M=" << M << ")" << std::endl;
    
    PolymerStats stats(Nmax);
    
    for (int trial = 0; trial < M; ++trial) {
        if ((trial + 1) % 100 == 0) {
            std::cout << "  Progress: " << (trial + 1) << "/" << M << "\r" << std::flush;
        }
        
        IdealPolymer polymer(dimension, trial);
        
        // Record initial configuration (N=2)
        for (int n = 2; n <= 2; ++n) {
            stats.R2_P[n] += polymer.getEndToEndDistanceSquared();
            stats.R2_g[n] += polymer.getRadiusOfGyrationSquared();
            stats.counts[n]++;
        }
        
        // Grow polymer and record at each step
        for (int n = 3; n <= Nmax; ++n) {
            polymer.grow();
            stats.R2_P[n] += polymer.getEndToEndDistanceSquared();
            stats.R2_g[n] += polymer.getRadiusOfGyrationSquared();
            stats.counts[n]++;
        }
    }
    
    std::cout << std::endl;
    
    // Write results to file
    std::ofstream file(filename);
    file << "N,R2_P,R2_g,count\n";
    for (int n = 2; n <= Nmax; ++n) {
        if (stats.counts[n] > 0) {
            file << n << ","
                 << stats.R2_P[n] / stats.counts[n] << ","
                 << stats.R2_g[n] / stats.counts[n] << ","
                 << stats.counts[n] << "\n";
        }
    }
    file.close();
    std::cout << "Results saved to " << filename << std::endl;
}

void simulateExcludedVolumePolymer(int dimension, int M, int Nmax, 
                                   const std::string& filename_bare,
                                   const std::string& filename_rosenbluth) {
    std::cout << "Simulating Excluded Volume Polymer (dim=" << dimension 
              << ", M=" << M << ")" << std::endl;
    
    PolymerStats stats_bare(Nmax);
    PolymerStats stats_rosenbluth(Nmax);
    
    for (int trial = 0; trial < M; ++trial) {
        if ((trial + 1) % 100 == 0) {
            std::cout << "  Progress: " << (trial + 1) << "/" << M << "\r" << std::flush;
        }
        
        ExcludedVolumePolymer polymer(dimension, trial);
        
        // Record initial configuration (N=2)
        double current_weight = 1.0;
        for (int w : polymer.getAllWeights()) {
            current_weight *= w;
        }
        
        stats_bare.R2_P[2] += polymer.getEndToEndDistanceSquared();
        stats_bare.R2_g[2] += polymer.getRadiusOfGyrationSquared();
        stats_bare.counts[2]++;
        
        stats_rosenbluth.R2_P[2] += current_weight * polymer.getEndToEndDistanceSquared();
        stats_rosenbluth.R2_g[2] += current_weight * polymer.getRadiusOfGyrationSquared();
        stats_rosenbluth.weights[2] += current_weight;
        stats_rosenbluth.counts[2]++;
        
        // Grow polymer and record at each step
        for (int n = 3; n <= Nmax; ++n) {
            bool success = polymer.grow();
            
            if (!success) {
                break; // Chain terminated
            }
            
            // Update weight
            current_weight *= polymer.getWeightAt(polymer.size() - 1);
            
            // Bare average
            stats_bare.R2_P[n] += polymer.getEndToEndDistanceSquared();
            stats_bare.R2_g[n] += polymer.getRadiusOfGyrationSquared();
            stats_bare.counts[n]++;
            
            // Rosenbluth weighted average
            stats_rosenbluth.R2_P[n] += current_weight * polymer.getEndToEndDistanceSquared();
            stats_rosenbluth.R2_g[n] += current_weight * polymer.getRadiusOfGyrationSquared();
            stats_rosenbluth.weights[n] += current_weight;
            stats_rosenbluth.counts[n]++;
        }
    }
    
    std::cout << std::endl;
    
    // Write bare average results
    std::ofstream file_bare(filename_bare);
    file_bare << "N,R2_P,R2_g,count\n";
    for (int n = 2; n <= Nmax; ++n) {
        if (stats_bare.counts[n] > 0) {
            file_bare << n << ","
                     << stats_bare.R2_P[n] / stats_bare.counts[n] << ","
                     << stats_bare.R2_g[n] / stats_bare.counts[n] << ","
                     << stats_bare.counts[n] << "\n";
        }
    }
    file_bare.close();
    std::cout << "Bare average results saved to " << filename_bare << std::endl;
    
    // Write Rosenbluth weighted results
    std::ofstream file_rosenbluth(filename_rosenbluth);
    file_rosenbluth << "N,R2_P,R2_g,total_weight,count\n";
    for (int n = 2; n <= Nmax; ++n) {
        if (stats_rosenbluth.counts[n] > 0 && stats_rosenbluth.weights[n] > 0) {
            file_rosenbluth << n << ","
                           << stats_rosenbluth.R2_P[n] / stats_rosenbluth.weights[n] << ","
                           << stats_rosenbluth.R2_g[n] / stats_rosenbluth.weights[n] << ","
                           << stats_rosenbluth.weights[n] << ","
                           << stats_rosenbluth.counts[n] << "\n";
        }
    }
    file_rosenbluth.close();
    std::cout << "Rosenbluth results saved to " << filename_rosenbluth << std::endl;
}

int main() {
    const int M = 1000;
    const int Nmax = 100;
    
    std::cout << "======================================" << std::endl;
    std::cout << "Monte Carlo Polymer Simulations" << std::endl;
    std::cout << "M = " << M << " realizations" << std::endl;
    std::cout << "Nmax = " << Nmax << " monomers" << std::endl;
    std::cout << "======================================\n" << std::endl;
    
    // Part A: Ideal Polymer
    std::cout << "PART A: IDEAL POLYMER\n" << std::endl;
    simulateIdealPolymer(2, M, Nmax, "ideal_2d.csv");
    std::cout << std::endl;
    simulateIdealPolymer(3, M, Nmax, "ideal_3d.csv");
    
    std::cout << "\n======================================\n" << std::endl;
    
    // Part B: Excluded Volume Polymer
    std::cout << "PART B: EXCLUDED VOLUME POLYMER\n" << std::endl;
    simulateExcludedVolumePolymer(2, M, Nmax, 
                                  "excluded_2d_bare.csv", 
                                  "excluded_2d_rosenbluth.csv");
    std::cout << std::endl;
    simulateExcludedVolumePolymer(3, M, Nmax, 
                                  "excluded_3d_bare.csv", 
                                  "excluded_3d_rosenbluth.csv");
    
    std::cout << "\n======================================" << std::endl;
    std::cout << "Simulation complete!" << std::endl;
    std::cout << "Run 'python visualize.py' to generate plots" << std::endl;
    std::cout << "======================================" << std::endl;
    
    return 0;
}