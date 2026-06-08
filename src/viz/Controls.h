#pragma once

/**
 * @file Controls.h
 * @brief Button-grid widget used for app-level commands.
 */

#include "viz/Theme.h"

#include <raylib.h>

#include <functional>
#include <string>
#include <vector>

namespace cnnv::viz {

/**
 * @brief Compact callback-driven command grid.
 *
 * The widget is stateless beyond button definitions and transient hover/press
 * indices. `App` wires each callback to a real action such as reset, flip,
 * setup, or load FEN.
 */
class Controls {
public:
    /** @brief Callback invoked when a button is pressed. */
    using Callback = std::function<void()>;

    /** @brief Predicate used to render a button as latched/on. */
    using ActivePredicate = std::function<bool()>;

    /** @brief Removes every button. */
    void clear() noexcept { m_buttons.clear(); }

    /** @brief Adds a labeled button, callback, and optional active state. */
    void addButton(std::string label, Callback onPress,
                   ActivePredicate isActive = {});

    /** @brief Sets the command-area rectangle. */
    void setBounds(Rectangle r) noexcept { m_bounds = r; }

    /** @brief Returns the command-area rectangle. */
    Rectangle bounds() const noexcept { return m_bounds; }

    /**
     * @brief Processes the most recent mouse input frame.
     *
     * Call once per frame before `draw()`.
     */
    void update();

    /** @brief Draws all buttons. */
    void draw(const Theme& theme = defaultTheme()) const;

private:
    /**
     * @brief Stored button label and command callback.
     */
    struct Button {
        std::string label;
        Callback onPress;
        ActivePredicate isActive;
    };

    Rectangle buttonRect(std::size_t index) const;

    Rectangle m_bounds{0, 0, 0, 0};
    std::vector<Button> m_buttons;
    int m_hoverIndex = -1;
    int m_pressedIndex = -1;
};

}  // namespace cnnv::viz
