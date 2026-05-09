#include "viz/Theme.h"

namespace cnnv::viz {

namespace {

Font fontOrDefault(Font candidate) {
    return candidate.texture.id != 0 ? candidate : GetFontDefault();
}

}  // namespace

const Theme& defaultTheme() noexcept {
    static const Theme t;
    return t;
}

void drawText(const Theme& theme, const char* text, int x, int y,
              int fontSize, Color tint) {
    Font f = fontOrDefault(theme.fontUi);
    DrawTextEx(f, text,
               Vector2{static_cast<float>(x), static_cast<float>(y)},
               static_cast<float>(fontSize), 1.0f, tint);
}

void drawTextMono(const Theme& theme, const char* text, int x, int y,
                  int fontSize, Color tint) {
    Font f = fontOrDefault(theme.fontMono);
    DrawTextEx(f, text,
               Vector2{static_cast<float>(x), static_cast<float>(y)},
               static_cast<float>(fontSize), 1.0f, tint);
}

int measureText(const Theme& theme, const char* text, int fontSize) {
    Font f = fontOrDefault(theme.fontUi);
    Vector2 sz = MeasureTextEx(f, text, static_cast<float>(fontSize), 1.0f);
    return static_cast<int>(sz.x);
}

int measureTextMono(const Theme& theme, const char* text, int fontSize) {
    Font f = fontOrDefault(theme.fontMono);
    Vector2 sz = MeasureTextEx(f, text, static_cast<float>(fontSize), 1.0f);
    return static_cast<int>(sz.x);
}

}  // namespace cnnv::viz
