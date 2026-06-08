#pragma once

/**
 * @file Theme.h
 * @brief Shared colors, fonts, and text drawing helpers for the UI.
 */

#include <raylib.h>

namespace cnnv::viz {

/**
 * @brief Shared visual constants for the raylib UI.
 *
 * Colors, fonts, and common UI styling live here so the look-and-feel can be
 * adjusted without editing every widget. Font fields may stay default-zero;
 * drawing helpers fall back to raylib's built-in font in that case.
 */
struct Theme {
    /** @brief Window background. */
    Color background        = Color{ 22,  22,  26, 255};

    /** @brief Light board-square color. */
    Color squareLight       = Color{238, 216, 192, 255};

    /** @brief Dark board-square color. */
    Color squareDark        = Color{171, 122, 101, 255};

    /** @brief Accent colors used for data and interaction state. */
    Color accentGreen       = Color{129, 171, 126, 230};
    Color accentRed         = Color{226,  92,  78, 235};
    Color accentBlue        = Color{ 96, 138, 178, 225};
    Color accentOrange      = Color{226, 102,  78, 235};
    Color accentMagenta     = Color{189,  86, 122, 230};
    Color accentYellow      = Color{255, 209,  90, 235};

    /**
     * @brief Per-architecture signature colors.
     *
     * Each network view carries one distinct signature hue used for its chrome
     * (header accent strip, active selector tab, mode-toggle active segment,
     * selection rings) so the four algorithms are visually distinguishable.
     * Signed-activation data keeps the semantic green/red diverging palette.
     */
    Color sigNnue           = Color{244, 168,  35, 255};  // amber
    Color sigCnn            = Color{ 26, 169, 160, 255};  // teal
    Color sigBt4            = Color{140,  92, 240, 255};  // violet
    Color sigClassical      = Color{224,  82, 156, 255};  // rose

    /** @brief Selected-square highlight. */
    Color selection         = Color{255, 209,  90, 200};

    /** @brief Last-move highlight. */
    Color lastMove          = Color{255, 209,  90, 110};

    /** @brief Legal quiet-move destination dot color. */
    Color legalDot          = Color{ 30,  30,  35, 110};

    /** @brief Legal capture destination ring color. */
    Color legalCaptureRing  = Color{ 30,  30,  35, 130};

    /** @brief King-in-check halo color. */
    Color checkWarning      = Color{226,  92,  78, 200};

    /** @brief Neutral activation-overlay color. */
    Color overlayZero       = Color{ 41,  43,  49, 220};

    /** @brief Positive/high activation-overlay color. */
    Color overlayHot        = Color{226, 102,  78, 230};

    /** @brief Negative/low activation-overlay color. */
    Color overlayCold       = Color{ 96, 138, 178,  90};

    /** @brief Panel and UI chrome colors. */
    Color panelBackground   = Color{ 28,  29,  33, 255};
    Color panelBorder       = Color{ 50,  52,  58, 255};
    Color textPrimary       = Color{231, 232, 234, 255};
    Color textMuted         = Color{160, 164, 170, 255};
    Color textDim           = Color{110, 114, 120, 255};
    Color buttonIdle        = Color{ 41,  43,  49, 255};
    Color buttonHover       = Color{ 56,  60,  68, 255};
    Color buttonActive      = Color{ 78,  84,  94, 255};

    /** @brief Coordinate label color around the board. */
    Color coordLabel        = Color{120, 110,  95, 220};

    /** @brief FEN strip colors. */
    Color fenStripBackground = Color{ 35,  37,  42, 255};
    Color fenStripText       = Color{205, 200, 188, 255};

    /**
     * @brief UI fonts.
     *
     * Default-zero `Font` values mean "use raylib built-in"; text helpers
     * handle the fallback.
     */
    Font fontUi   {};
    Font fontMono {};
};

/**
 * @brief Global default theme.
 */
const Theme& defaultTheme() noexcept;

/**
 * @brief Draws UI text using the theme font or raylib fallback.
 */
void drawText(const Theme& theme, const char* text, int x, int y,
              int fontSize, Color tint);

/**
 * @brief Draws monospace text using the theme font or raylib fallback.
 */
void drawTextMono(const Theme& theme, const char* text, int x, int y,
                  int fontSize, Color tint);

/**
 * @brief Measures UI text width.
 */
int  measureText(const Theme& theme, const char* text, int fontSize);

/**
 * @brief Measures monospace text width.
 */
int  measureTextMono(const Theme& theme, const char* text, int fontSize);

}  // namespace cnnv::viz
