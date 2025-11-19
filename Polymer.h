#ifndef POLYMER_H
#define POLYMER_H

#include <vector>
#include <array>
#include <random>
#include <set>
#include <cmath>

// 3D position structure
struct Position
{
    int x, y, z;

    Position(int x_ = 0, int y_ = 0, int z_ = 0) : x(x_), y(y_), z(z_) {}

    bool operator<(const Position &other) const
    {
        if (x != other.x)
            return x < other.x;
        if (y != other.y)
            return y < other.y;
        return z < other.z;
    }

    bool operator==(const Position &other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }

    Position operator+(const Position &other) const
    {
        return Position(x + other.x, y + other.y, z + other.z);
    }

    double distanceSquared(const Position &other) const
    {
        double dx = x - other.x;
        double dy = y - other.y;
        double dz = z - other.z;
        return dx * dx + dy * dy + dz * dz;
    }
};

// Base Polymer class
class Polymer
{
protected:
    std::vector<Position> monomers;
    int dimension;
    std::mt19937 rng;

public:
    Polymer(int dim, unsigned int seed = std::random_device{}())
        : dimension(dim), rng(seed)
    {
        monomers.clear();
        monomers.push_back(Position(0, 0, 0));
        monomers.push_back(Position(1, 0, 0));
    }

    virtual ~Polymer() = default;

    int size() const { return monomers.size(); }

    double getEndToEndDistanceSquared() const
    {
        return monomers.front().distanceSquared(monomers.back());
    }

    double getRadiusOfGyrationSquared() const
    {
        Position cm = getCenterOfMass();
        double rg2 = 0.0;
        for (const auto &pos : monomers)
        {
            rg2 += pos.distanceSquared(cm);
        }
        return rg2 / monomers.size();
    }

    Position getCenterOfMass() const
    {
        double cx = 0, cy = 0, cz = 0;
        for (const auto &pos : monomers)
        {
            cx += pos.x;
            cy += pos.y;
            cz += pos.z;
        }
        int n = monomers.size();
        return Position(std::round(cx / n), std::round(cy / n), std::round(cz / n));
    }

    virtual bool grow() = 0;
};

// Ideal Polymer - Random Walk
class IdealPolymer : public Polymer
{
private:
    std::vector<Position> getDirections() const
    {
        if (dimension == 2)
        {
            return {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}};
        }
        else
        {
            return {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
        }
    }

public:
    IdealPolymer(int dim, unsigned int seed = std::random_device{}())
        : Polymer(dim, seed) {}

    bool grow() override
    {
        auto directions = getDirections();
        std::uniform_int_distribution<> dis(0, directions.size() - 1);
        Position lastPos = monomers.back();
        Position newPos = lastPos + directions[dis(rng)];
        monomers.push_back(newPos);
        return true;
    }
};

// Excluded Volume Polymer - Self-Avoiding Walk
class ExcludedVolumePolymer : public Polymer
{
private:
    std::set<Position> occupied;
    std::vector<int> weights;

    std::vector<Position> getDirections() const
    {
        if (dimension == 2)
        {
            return {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}};
        }
        else
        {
            return {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
        }
    }

public:
    ExcludedVolumePolymer(int dim, unsigned int seed = std::random_device{}())
        : Polymer(dim, seed)
    {
        occupied.insert(monomers[0]);
        occupied.insert(monomers[1]);
        weights.clear();
        weights.push_back(1);
        weights.push_back(dimension == 2 ? 3 : 5);
    }

    bool grow() override
    {
        auto directions = getDirections();
        Position lastPos = monomers.back();

        // Find available positions
        std::vector<Position> available;
        for (const auto &dir : directions)
        {
            Position candidate = lastPos + dir;
            if (occupied.find(candidate) == occupied.end())
            {
                available.push_back(candidate);
            }
        }

        if (available.empty())
        {
            weights.push_back(0);
            return false; // Chain terminated
        }

        // Choose randomly from available positions
        std::uniform_int_distribution<> dis(0, available.size() - 1);
        Position newPos = available[dis(rng)];

        monomers.push_back(newPos);
        occupied.insert(newPos);
        weights.push_back(available.size());

        return true;
    }

    double getWeight() const
    {
        double w = 1.0;
        for (int m : weights)
        {
            if (m == 0)
                return 0.0;
            w *= m;
        }
        return w;
    }

    int getWeightAt(int n) const
    {
        if (n < 0 || n >= weights.size())
            return 0;
        return weights[n];
    }

    std::vector<int> getAllWeights() const
    {
        return weights;
    }
};

#endif // POLYMER_H