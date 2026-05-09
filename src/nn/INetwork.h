#pragma once

/**
 * @file INetwork.h
 * @brief Common interface implemented by all neural-network evaluators.
 */

#include <string>

namespace cnnv::chess {
class Position;
}

namespace cnnv::nn {

class ActivationSnapshot;

/**
 * @brief Polymorphic evaluator interface for supported NN architectures.
 *
 * The application can swap between NNUE, LC0 CNN, BT4, or future backends
 * through this interface while the UI consumes only activation snapshots and
 * display names.
 */
class INetwork {
   public:
    virtual ~INetwork() = default;

    /**
     * @brief Loads weights or model metadata from disk.
     * @param path Backend-specific weight path.
     */
    virtual void load(const std::string& path) = 0;

    /**
     * @brief Evaluates a chess position and fills a visualization snapshot.
     * @param pos Position to evaluate.
     * @param out Snapshot destination. Implementations replace or append their
     * own architecture keys.
     */
    virtual void evaluate(const cnnv::chess::Position& pos,
                          ActivationSnapshot& out) const = 0;

    /**
     * @brief Human-readable network name shown in the UI.
     */
    virtual std::string name() const = 0;
};

}  // namespace cnnv::nn
