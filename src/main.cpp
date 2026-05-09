#include "viz/App.h"

int main(int argc, char** argv) {
    const char* configPath = (argc > 1) ? argv[1] : "config.ini";
    cnnv::viz::App app(configPath);
    return app.run();
}
