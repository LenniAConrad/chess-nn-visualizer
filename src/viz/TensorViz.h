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
 */
void drawSectionHeader(Rectangle bounds,
                       const char* title,
                       const char* detail,
                       const Theme& theme);

/**
 * @brief Draws and updates a compact/detailed mode toggle.
 * @return Updated detailed-state value.
 */
bool drawModeToggle(Rectangle bounds, bool detailed, const Theme& theme);

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
