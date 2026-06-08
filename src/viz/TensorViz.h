#pragma once

/**
 * @file TensorViz.h
 * @brief Shared drawing helpers for tensor previews, charts, and inspectors.
 */

#include "viz/Theme.h"

#include <raylib.h>

#include <cstddef>
#include <string>
#include <vector>

namespace cnnv::viz::tensorviz {

/**
 * @brief Basic statistics for a flat tensor used by visualization widgets.
 */
struct TensorStats {
    std::size_t count = 0;
    float min = 0.0f;
    float max = 0.0f;
    float mean = 0.0f;
    float meanAbs = 0.0f;
    float rms = 0.0f;
    float maxAbs = 0.0f;
};

/**
 * @brief Computes summary statistics for flat tensor values.
 */
TensorStats computeStats(const std::vector<float>& values);

/**
 * @brief Gibt den groessten Absolutwert eines flachen Tensors zurueck.
 *
 * Wird von mehreren Netzwerkansichten fuer Heatmap-Skalierung genutzt, damit
 * jede Ansicht nicht dieselbe Schleife lokal duplizieren muss.
 */
float maxAbs(const std::vector<float>& values);

/**
 * @brief Liefert den algebraischen Namen eines LERF-Squares.
 *
 * Gueltige Indizes sind 0..63 mit a1 == 0. Ungueltige Indizes liefern "??",
 * was fuer Debug- und Fallback-Beschriftungen stabil bleibt.
 */
const char* squareName(int square);

/**
 * @brief Kuerzt Text pixelgenau auf die angegebene Breite.
 *
 * Nutzt die aktuelle Theme-Schriftmessung und haengt "..." an, sobald der Text
 * nicht mehr in den Zielbereich passt. @p mono waehlt die Monospace-Messung.
 */
std::string fitText(const Theme& theme,
                    std::string text,
                    int fontSize,
                    bool mono,
                    float maxWidth);

/**
 * @brief Berechnet das Rechteck eines LERF-Squares innerhalb eines Brettes.
 *
 * LERF nutzt a1 == 0. Mit @p whiteDown liegt Weiss unten; andernfalls wird die
 * Ansicht fuer Schwarz gespiegelt. Der Helfer haelt Board-Overlays konsistent.
 */
Rectangle lerfSquare(Rectangle board, int square, bool whiteDown = true);

/**
 * @brief Zeichnet ein neutrales 8x8-Minibrett ohne Figuren.
 *
 * Die Netzwerkansichten nutzen diesen gemeinsamen Hintergrund fuer Attention-
 * und Aktivierungs-Overlays.
 */
void drawMiniBoard(Rectangle board, const Theme& theme);

/**
 * @brief Zeichnet Text zentriert in einem Rechteck.
 *
 * @p mono waehlt Monospace-Messung und -Rendering, normale Labels nutzen die
 * Standardschrift des Themes.
 */
void drawCenteredText(const Theme& theme,
                      const char* text,
                      Rectangle bounds,
                      int fontSize,
                      Color color,
                      bool mono = false);

/**
 * @brief Registers the 2D camera currently transforming an activation view.
 *
 * While a camera is active, panelMousePosition() returns the cursor mapped into
 * the view's world space so interactive hit-testing stays correct under
 * zoom/pan. App sets this around the active view's draw and clears it after.
 */
void setActiveCamera(const Camera2D& camera);

/** @brief Clears any camera registered by setActiveCamera(). */
void clearActiveCamera();

/**
 * @brief Returns the cursor position in active-view coordinates.
 *
 * Equivalent to GetMousePosition() when no camera is active; otherwise the
 * screen cursor mapped through the active camera (GetScreenToWorld2D).
 */
Vector2 panelMousePosition();

/**
 * @brief Linearly blends two colors.
 */
Color blend(Color a, Color b, float t);

/**
 * @brief Returns a copy of a color with replaced alpha.
 */
Color withAlpha(Color c, unsigned char alpha);

/**
 * @brief Maps a signed activation value to the theme's diverging palette.
 */
Color signedColor(float value, float scale, const Theme& theme);

/**
 * @brief Draws a titled section header with optional detail text.
 *
 * When @p accent has non-zero alpha a signature-colored strip is drawn on the
 * header's left edge; otherwise the header uses the neutral theme chrome.
 */
void drawSectionHeader(Rectangle bounds,
                       const char* title,
                       const char* detail,
                       const Theme& theme,
                       Color accent = Color{0, 0, 0, 0});

/**
 * @brief Draws and updates a compact/detailed mode toggle.
 *
 * When @p accent has non-zero alpha the active segment is tinted with it.
 * @return Updated detailed-state value.
 */
bool drawModeToggle(Rectangle bounds, bool detailed, const Theme& theme,
                    Color accent = Color{0, 0, 0, 0});

/**
 * @brief Three-way view-mode selector: abstract / detailed / diagram.
 *
 * Segments are equal thirds; the active segment is tinted with @p accent when
 * its alpha is non-zero.
 * @param mode Current mode (0 = abstract, 1 = detailed, 2 = diagram).
 * @return Updated mode index.
 */
int drawModeSelector(Rectangle bounds, int mode, const Theme& theme,
                     Color accent = Color{0, 0, 0, 0});

/**
 * @brief Five-segment engine-lab view-mode switcher.
 *
 * Segments: Overview / Trace / All / Atlas / Diagram (modes 0..4). The active
 * segment is tinted with @p accent. Mirrors chess-rtk's network-view mode
 * switcher; shared by all the activation views so they look identical.
 * @return Updated mode index.
 */
int drawModeSwitcher5(Rectangle bounds, int mode, Color accent,
                      const Theme& theme);

/**
 * @brief Draws a summarized network block in overview mode.
 */
void drawAbstractBlock(Rectangle bounds,
                       const std::string& title,
                       const std::string& detail,
                       float activity,
                       Color accent,
                       bool selected,
                       const Theme& theme);

/**
 * @brief Draws one layer card with an embedded tensor preview.
 */
void drawLayerCard(Rectangle bounds,
                   const std::string& name,
                   const std::string& shape,
                   const std::vector<float>& values,
                   const std::vector<std::size_t>& shapeDims,
                   float rmsScale,
                   Color accent,
                   bool selected,
                   bool dimmed,
                   const Theme& theme);

/**
 * @brief Draws a compact tensor heatmap or bar preview.
 */
void drawTensorPreview(const std::vector<float>& values,
                       const std::vector<std::size_t>& shapeDims,
                       Rectangle bounds,
                       const Theme& theme,
                       float scale = 0.0f,
                       bool highDetail = false);

/**
 * @brief Draws detailed statistics and preview for one selected tensor.
 */
void drawInspector(Rectangle bounds,
                   const std::string& title,
                   const std::string& shape,
                   const std::vector<float>& values,
                   const std::vector<std::size_t>& shapeDims,
                   const std::string& note,
                   const Theme& theme);

/**
 * @brief Draws a histogram over tensor values.
 */
void drawHistogram(Rectangle bounds,
                   const std::vector<float>& values,
                   float scale,
                   const Theme& theme);

/**
 * @brief Draws a diverging color legend for signed activations.
 */
void drawDivergingLegend(Rectangle bounds,
                         float scale,
                         const Theme& theme);

/**
 * @brief Draws labeled metric bars sharing one scale.
 */
void drawMetricBars(Rectangle bounds,
                    const std::vector<std::string>& labels,
                    const std::vector<float>& values,
                    float scale,
                    const Theme& theme);

/**
 * @brief Draws an elbow connector between two layer rectangles.
 */
void drawElbowConnection(Rectangle from,
                         Rectangle to,
                         Color color,
                         bool active,
                         bool dimmed,
                         const Theme& theme);

/**
 * @brief Draws sampled weighted connections between two tensor previews.
 */
void drawNodeFanConnections(const std::vector<float>& fromValues,
                            const std::vector<std::size_t>& fromShape,
                            Rectangle fromBounds,
                            const std::vector<float>& toValues,
                            const std::vector<std::size_t>& toShape,
                            Rectangle toBounds,
                            Color color,
                            std::size_t maxEdges,
                            const Theme& theme);

}  // namespace cnnv::viz::tensorviz
