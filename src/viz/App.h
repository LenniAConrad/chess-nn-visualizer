#pragma once

/**
 * @file App.h
 * @brief raylib application coordinator for the chess and NN visualizer UI.
 */

#include "chess/Position.h"
#include "game/Game.h"
#include "io/ConfigIo.h"
#include "nn/ActivationSnapshot.h"
#include "nn/lc0_bt4/Bt4Network.h"
#include "nn/lc0_cnn/Lc0CnnNetwork.h"
#include "nn/nnue/NnueNetwork.h"
#include "viz/BoardView.h"
#include "viz/Bt4View.h"
#include "viz/Controls.h"
#include "viz/CnnView.h"
#include "viz/EditorMode.h"
#include "viz/EditorPalette.h"
#include "viz/EditorPanel.h"
#include "viz/FenInputDialog.h"
#include "viz/MoveListView.h"
#include "viz/NnueView.h"
#include "viz/PieceSprites.h"
#include "viz/PromotionPicker.h"
#include "viz/Theme.h"

#include <atomic>
#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

namespace cnnv::viz {

/**
 * @brief Top-level coordinator for the interactive visualizer.
 *
 * `App` owns the live `Game`, raylib-backed assets, board view, command
 * controls, editor widgets, activation panels, async evaluation jobs, and
 * search preview state. It is the single owner of raylib window lifetime.
 */
class App {
public:
    /**
     * @brief Creates the application and reads runtime configuration.
     * @param configPath Path to the flat config file.
     */
    explicit App(const std::string& configPath = "config.ini");

    /**
     * @brief Releases any raylib resources still owned by the app.
     */
    ~App();

    App(const App&) = delete;
    App& operator=(const App&) = delete;

    /**
     * @brief Runs the main event/render loop.
     * @return Process exit code.
     */
    int run();

private:
    /**
     * @brief Neural-network architecture selected for activation rendering.
     */
    enum class Arch { Nnue, Cnn, Bt4, Off };

    /** @brief Number of architecture selector buttons. */
    static constexpr std::size_t kArchCount = 4;

    /**
     * @brief Result object returned from an asynchronous network evaluation.
     */
    struct EvalResult {
        std::uint64_t requestId = 0;
        std::uint64_t hash = 0;
        Arch arch = Arch::Nnue;
        cnnv::chess::Color sideToMove = cnnv::chess::Color::White;
        cnnv::nn::ActivationSnapshot snapshot;
        bool ok = true;
        bool hasCentipawns = false;
        bool hasWdl = false;
        float centipawns = 0.0f;
        std::array<float, 3> wdl{0.0f, 0.0f, 0.0f};
        std::string error;
    };

    /**
     * @brief Cached network evaluation for one `(arch, position hash)` pair.
     */
    struct EvalCache {
        cnnv::nn::ActivationSnapshot snapshot;
        std::uint64_t hash = 0;
        Arch arch = Arch::Nnue;
        cnnv::chess::Color sideToMove = cnnv::chess::Color::White;
        bool have = false;
        bool hasCentipawns = false;
        bool hasWdl = false;
        float centipawns = 0.0f;
        std::array<float, 3> wdl{0.0f, 0.0f, 0.0f};
        std::uint64_t lastUsed = 0;
    };

    /**
     * @brief Lightweight search-line preview generated off the render thread.
     */
    struct SearchPreview {
        cnnv::chess::Position position;
        cnnv::chess::Move lastMove = cnnv::chess::Move::none();
        std::string line;
        std::uint64_t nodes = 0;
        std::uint64_t serial = 0;
        int depth = 0;
        float score = 0.0f;
        bool have = false;
        bool complete = false;
    };

    /** @brief Creates command buttons and wires callbacks. */
    void buildControls();

    /** @brief Computes all widget rectangles for the current window size. */
    void layout();

    /** @brief Processes keyboard and mouse input for one frame. */
    void handleInput();

    /** @brief Draws the complete frame. */
    void render();

    /** @brief Resets the game to standard startpos and clears transient UI. */
    void resetGame();

    /** @brief Toggles board orientation. */
    void flipBoard();

    /** @brief Undoes one move when possible. */
    void undoMove();

    /** @brief Redoes one move when possible. */
    void redoMove();

    /** @brief Handles repeated keyboard navigation through move history. */
    void handleHistoryNavigationKeys();

    /** @brief Resets key-repeat state for history navigation. */
    void resetHistoryNavigationRepeat() noexcept;

    /** @brief Applies a random legal move for quick interaction demos. */
    void playRandomMove();

    /** @brief Saves the current position to the configured FEN file. */
    void saveCurrentFen();

    /** @brief Opens or consumes the FEN dialog depending on current state. */
    void loadFenFromDialog();

    /** @brief Loads UI fonts from configured assets. */
    void loadFonts();

    /** @brief Releases UI font resources. */
    void unloadFonts();

    /** @brief Applies a named board/theme palette from configuration. */
    void applyBoardPalette(const std::string& paletteName);

    /** @brief Switches from play mode into setup/editor mode. */
    void enterEditor();

    /** @brief Leaves editor mode, optionally committing the edited position. */
    void leaveEditor(bool commit);

    /** @brief Applies an editor board click to the current brush. */
    void handleEditorBoardClick(cnnv::chess::Square sq, bool isLeft);

    /** @brief Toggles fullscreen and refreshes layout sizing. */
    void toggleFullscreen();

    /** @brief Synchronizes cached window dimensions with raylib state. */
    void syncWindowSize();

    /** @brief Applies a legal move and updates the game clock. */
    bool tryMoveWithClock(cnnv::chess::Move m);

    /** @brief Restores both clocks to configured initial time. */
    void resetClock();

    /** @brief Decrements the active side's clock for the current frame. */
    void updateClock();

    /** @brief Adds increment to the side that just moved. */
    void addClockIncrement(cnnv::chess::Color side);

    /** @brief Formats the status banner, including game and clock state. */
    std::string statusText() const;

    /** @brief Formats seconds as a chess-clock string. */
    std::string formatClock(double seconds) const;

    /** @brief Draws the clock widget inside the status area. */
    void drawClockArea(Rectangle statusBounds) const;

    /** @brief Selects one activation architecture and invalidates UI state. */
    void selectArchitecture(Arch arch);

    /** @brief Handles architecture-selector click/hover state. */
    void updateArchitectureSelector();

    /** @brief Draws the architecture selector tabs. */
    void drawArchitectureSelector() const;

    /** @brief Mutable active activation view, or null when disabled. */
    IActivationView* activeActivationView() noexcept;

    /** @brief Const active activation view, or null when disabled. */
    const IActivationView* activeActivationView() const noexcept;

    /** @brief Starts or stops the lightweight search preview. */
    void toggleSearch();

    /** @brief Launches the background search preview job. */
    void startSearch();

    /** @brief Requests search cancellation and joins as needed. */
    void stopSearch();

    /** @brief Collects completed search work without blocking the frame. */
    void pollSearchJob();

    /** @brief Returns the latest search preview under lock. */
    SearchPreview currentSearchPreview() const;

    /** @brief Returns the preview currently acknowledged by the renderer. */
    SearchPreview currentDisplayedSearchPreview() const;

    /** @brief Tests whether a search preview is ready for display. */
    bool searchPreviewReadyForDisplayAck(const SearchPreview& search) const;

    /** @brief Promotes a preview serial to the displayed state. */
    bool promoteSearchPreviewForDisplay(std::uint64_t serial);

    /** @brief Marks a preview serial as consumed by the frame renderer. */
    void markSearchPreviewDisplayed(std::uint64_t serial);

    /** @brief Position currently used as the evaluation root. */
    cnnv::chess::Position currentEvalPosition() const;

    /** @brief Zobrist hash for `currentEvalPosition()`. */
    std::uint64_t currentEvalHash() const;

    /** @brief Publishes one search preview from the background worker. */
    void publishSearchPreview(cnnv::chess::Position position,
                              cnnv::chess::Move lastMove,
                              std::string line,
                              int depth,
                              float score,
                              std::uint64_t nodes,
                              bool complete);

    int m_width  = 1280;
    int m_height = 800;
    std::string m_title = "cpp-nn-visualizer";

    cnnv::io::Config m_config;

    Theme m_theme;
    std::unique_ptr<PieceSprites> m_sprites;
    cnnv::game::Game m_game;
    std::unique_ptr<BoardView> m_board;
    Controls m_controls;
    std::unique_ptr<PromotionPicker> m_promotionPicker;
    FenInputDialog m_fenDialog;
    std::unique_ptr<MoveListView> m_moveList;
    std::unique_ptr<NnueView> m_nnueView;
    std::unique_ptr<CnnView> m_cnnView;
    std::unique_ptr<Bt4View> m_bt4View;
    std::array<Rectangle, kArchCount> m_archButtons{};
    Rectangle m_activationPanelBounds{0, 0, 0, 0};

    EditorMode m_editor;
    std::unique_ptr<EditorPalette> m_editorPalette;
    std::unique_ptr<EditorPanel>   m_editorPanel;
    bool m_fenDialogOpensEditor = false;

    std::optional<cnnv::chess::Square> m_selected;
    cnnv::chess::Bitboard m_legalTargets = 0;

    /**
     * @brief Drag-and-drop state for piece movement.
     *
     * `m_dragFrom` is set on press over the side-to-move's piece. `m_dragging`
     * flips true only after the cursor crosses a small movement threshold so a
     * simple click remains a click, not a zero-distance drag.
     */
    std::optional<cnnv::chess::Square> m_dragFrom;
    Vector2 m_dragStartMouse{0.0f, 0.0f};
    bool m_dragging = false;

    int m_historyRepeatDirection = 0;
    double m_historyRepeatNextAt = 0.0;

    /**
     * @brief Network evaluators and cached snapshot state.
     *
     * Evaluation is launched only when the live position hash or selected
     * architecture changes. Jobs run off the render thread so heavy CNN
     * positions do not stall input.
     */
    cnnv::nn::nnue::Network m_nnue;
    cnnv::nn::lc0_cnn::Network m_cnn;
    cnnv::nn::lc0_bt4::Network m_bt4;
    bool m_cnnLoaded = false;
    Arch m_arch = Arch::Nnue;

    bool m_clockEnabled = false;
    double m_clockInitialSeconds = 300.0;
    double m_clockIncrementSeconds = 0.0;
    double m_clockWhiteSeconds = 300.0;
    double m_clockBlackSeconds = 300.0;
    double m_clockLastUpdate = 0.0;
    std::optional<cnnv::chess::Color> m_clockExpiredSide;

    cnnv::nn::ActivationSnapshot m_snapshot;
    std::uint64_t m_lastEvalHash = 0;
    Arch m_lastEvalArch = Arch::Nnue;
    bool m_haveEval = false;
    float m_lastCentipawns = 0.0f;
    float m_lastWdl[3] = {0.0f, 0.0f, 0.0f};
    cnnv::chess::Color m_lastEvalSideToMove = cnnv::chess::Color::White;
    std::array<std::unordered_map<std::uint64_t, EvalCache>, kArchCount> m_evalCaches{};
    std::uint64_t m_evalCacheClock = 0;
    std::string m_nnueStatus;
    std::string m_cnnStatus;
    mutable std::mutex m_networkEvalMutex;

    /** @brief Recomputes the board heatmap overlay from the active snapshot. */
    void refreshActivationOverlay();

    /** @brief Starts or reuses evaluation when the current position is stale. */
    void evaluateIfStale();

    /** @brief Checks the async evaluation future without blocking the frame. */
    void pollEvalJob();

    /** @brief Launches one asynchronous evaluation job. */
    void startEvalJob(cnnv::chess::Position pos, std::uint64_t hash, Arch arch);

    /** @brief Applies a completed evaluation result to UI state and caches. */
    void applyEvalResult(EvalResult&& result);

    /** @brief Scores one search position using the requested architecture. */
    int evaluateSearchPositionForArch(const cnnv::chess::Position& pos,
                                      Arch arch,
                                      bool cnnLoaded) const;

    /** @brief Looks up a cached evaluation by architecture and hash. */
    EvalCache* cachedEval(Arch arch, std::uint64_t hash);

    /** @brief Stores a completed evaluation and returns its cache slot. */
    EvalCache& storeEvalResult(EvalResult&& result);

    /** @brief Enforces per-architecture evaluation-cache limits. */
    void trimEvalCache(Arch arch);

    /** @brief Cache-entry limit for one architecture. */
    std::size_t evalCacheLimit(Arch arch) const noexcept;

    /** @brief Displays a cached evaluation without launching a new job. */
    void displayCachedEval(const EvalCache& cache, bool restartFade);

    /** @brief Blocks until the current evaluation job has finished. */
    void waitForEvalJob();

    /** @brief Tests whether a job result still matches the current request. */
    bool evalJobTargetsCurrent(std::uint64_t hash, Arch arch) const noexcept;

    /** @brief Tests whether the in-flight job already targets current state. */
    bool evalInFlightForCurrent() const noexcept;

    /** @brief Attempts to load NNUE weights and records user-facing status. */
    bool tryLoadNnue(const std::string& path);

    /** @brief Attempts to load CNN weights and records user-facing status. */
    bool tryLoadCnn(const std::string& path);

    /** @brief Attempts to load BT4 visual-model metadata. */
    bool tryLoadBt4(const std::string& path);

    /** @brief Cycles to the next available architecture. */
    void cycleArchitecture();

    /** @brief Short architecture label for UI text. */
    const char* archName(Arch a) const noexcept;

    bool m_fontsLoaded = false;

    std::future<EvalResult> m_evalFuture;
    bool m_evalInFlight = false;
    std::uint64_t m_evalRequestSerial = 0;
    std::uint64_t m_evalInFlightRequestId = 0;
    std::uint64_t m_evalInFlightHash = 0;
    Arch m_evalInFlightArch = Arch::Nnue;
    bool m_evalFailed = false;
    std::uint64_t m_evalFailedHash = 0;
    Arch m_evalFailedArch = Arch::Nnue;
    std::string m_evalError;
    float m_activationFade = 1.0f;

    std::future<void> m_searchFuture;
    std::atomic<bool> m_searchCancel{false};
    mutable std::mutex m_searchMutex;
    std::condition_variable m_searchDisplayCv;
    SearchPreview m_searchPreview;
    SearchPreview m_searchDisplayedPreview;
    bool m_searchActive = false;
    bool m_searchRunning = false;
    std::uint64_t m_searchRootHash = 0;
    std::uint64_t m_searchPreviewSerial = 0;
    std::uint64_t m_searchDisplayedSerial = 0;
};

}  // namespace cnnv::viz
