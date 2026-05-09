#pragma once

/**
 * @file EditorPanel.h
 * @brief Side-panel controls for custom-position setup mode.
 */

#include "viz/EditorMode.h"
#include "viz/Theme.h"

#include <raylib.h>

namespace cnnv::viz {

/**
 * @brief Right-side controls for editing non-piece position state.
 *
 * The panel owns no chess state; it mutates shared `EditorMode` directly.
 * `takeAction()` is its outward signal for Validate, Apply, and Cancel button
 * presses.
 */
class EditorPanel {
public:
    /** @brief One-shot action requested by a panel button. */
    enum class Action { None, Validate, Apply, Cancel };

    /** @brief Creates a panel bound to shared editor state. */
    explicit EditorPanel(EditorMode& mode) : m_mode(&mode) {}

    /** @brief Sets panel bounds. */
    void setBounds(Rectangle r) noexcept { m_bounds = r; }

    /** @brief Returns panel bounds. */
    Rectangle bounds() const noexcept { return m_bounds; }

    /** @brief Processes mouse input and updates pending action. */
    void update();

    /** @brief Draws editor state controls and validation text. */
    void draw(const Theme& theme) const;

    /**
     * @brief Retrieves and clears the pending panel action.
     */
    Action takeAction() noexcept {
        Action a = m_pending;
        m_pending = Action::None;
        return a;
    }

private:
    struct LayoutY {
        float side;
        float castling;
        float ep;
        float halfmove;
        float fullmove;
        float fenLabel;
        float fenText;
        float issuesLabel;
        float buttonsBottom;
    };

    LayoutY layout() const noexcept;

    Rectangle sideButton(int idx) const;     // 0 = white, 1 = black
    Rectangle castleButton(int idx) const;   // 0..3: K Q k q
    Rectangle epPrevButton() const;
    Rectangle epNextButton() const;
    Rectangle halfMinusButton() const;
    Rectangle halfPlusButton() const;
    Rectangle fullMinusButton() const;
    Rectangle fullPlusButton() const;
    Rectangle validateButton() const;
    Rectangle applyButton() const;
    Rectangle cancelButton() const;

    void cycleEp(int direction);

    EditorMode* m_mode;
    Rectangle m_bounds{0, 0, 0, 0};
    Action m_pending = Action::None;
};

}  // namespace cnnv::viz
