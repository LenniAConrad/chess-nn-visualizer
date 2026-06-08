#pragma once

/**
 * @file IActivationView.h
 * @brief Interface for architecture-specific activation panels.
 */

#include "viz/Theme.h"

#include <raylib.h>

#include <string>

namespace cnnv::nn {
class ActivationSnapshot;
}

namespace cnnv::viz {

/**
 * @brief Abstract base for per-architecture activation panels.
 *
 * Implementations read from `ActivationSnapshot` string keys and render into a
 * fixed rectangle inside the right-side UI column.
 */
class IActivationView {
   public:
    virtual ~IActivationView() = default;

    /**
     * @brief Refreshes internal cached state from the latest snapshot.
     *
     * Implementations should copy only what they need because the snapshot may
     * be re-populated before the next frame.
     */
    virtual void update(const cnnv::nn::ActivationSnapshot& snap) = 0;

    /** @brief Sets panel bounds. */
    virtual void setBounds(Rectangle r) = 0;

    /** @brief Renders the panel. */
    virtual void draw(const Theme& theme = defaultTheme()) const = 0;

    /** @brief Short label for the architecture switcher. */
    virtual std::string name() const = 0;

    /**
     * @brief Sets this view's per-architecture signature color.
     *
     * Used for the header accent strip, mode-toggle active segment, and
     * selection highlights so each architecture is visually distinct.
     */
    virtual void setSignature(Color c) { m_signature = c; }

   protected:
    /** @brief Per-architecture signature color (neutral by default). */
    Color m_signature = Color{140, 144, 152, 255};
};

}  // namespace cnnv::viz
