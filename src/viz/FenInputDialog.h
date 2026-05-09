#pragma once

/**
 * @file FenInputDialog.h
 * @brief Modal single-line FEN input and validation widget.
 */

#include "chess/Position.h"
#include "viz/Theme.h"

#include <raylib.h>

#include <optional>
#include <string>

namespace cnnv::viz {

/**
 * @brief Single-line modal text input for FEN strings.
 *
 * While active, captures keyboard input, validates with `Fen::parse()`, and
 * returns the parsed position when Enter is pressed on a valid FEN. Escape
 * cancels; Backspace removes one character.
 */
class FenInputDialog {
public:
    FenInputDialog() = default;

    /** @brief Opens the dialog with optional initial text. */
    void open(const std::string& initial = "");

    /** @brief Closes the dialog without returning a position. */
    void close() noexcept { m_active = false; }

    /** @brief True while the dialog is visible and capturing input. */
    bool active() const noexcept { return m_active; }

    /** @brief Sets dialog bounds. */
    void setBounds(Rectangle r) noexcept { m_bounds = r; }

    /**
     * @brief Processes one input frame.
     * @return Parsed position exactly once when the user confirms valid FEN.
     */
    std::optional<cnnv::chess::Position> update();

    /** @brief Draws the modal dialog. */
    void draw(const Theme& theme = defaultTheme()) const;

private:
    Rectangle m_bounds{0, 0, 0, 0};
    std::string m_buffer;
    std::string m_errorMessage;
    bool m_active = false;
};

}  // namespace cnnv::viz
