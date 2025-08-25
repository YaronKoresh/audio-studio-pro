#!/bin/sh

info() {
    printf '\033[1;34m[INFO]\033[0m %s\n' "$1"
}

error() {
    printf '\033[1;31m[ERROR]\033[0m %s\n' "$1"
}

success() {
    printf '\033[1;32m[SUCCESS]\033[0m %s\n' "$1"
}

ensure_sudo() {
    if [ "$(id -u)" -ne 0 ]; then
        info "Administrator privileges are required. Requesting with sudo..."
        sudo sh "$0" --sudo-run
        exit $?
    fi
}

install_python() {
    info "Python3 not found. Attempting to install it..."
    ensure_sudo

    case "$(uname)" in
        Linux)
            if command -v apt-get >/dev/null 2>&1; then
                info "Debian/Ubuntu based system detected. Using apt-get..."
                apt-get update && apt-get install -y python3 python3-pip python3-venv
            elif command -v dnf >/dev/null 2>&1; then
                info "Fedora/RHEL based system detected. Using dnf..."
                dnf install -y python3 python3-pip
            elif command -v yum >/dev/null 2>&1; then
                info "CentOS/RHEL based system detected. Using yum..."
                yum install -y python3 python3-pip
            else
                error "Unsupported Linux distribution. Please install Python 3 and pip manually."
                exit 1
            fi
            ;;
        Darwin)
            info "macOS detected."
            if ! command -v brew >/dev/null 2>&1; then
                error "Homebrew not found. Please install it first from https://brew.sh"
                exit 1
            fi
            info "Using Homebrew to install Python..."
            brew install python
            ;;
        *)
            error "Unsupported operating system: $(uname). Please install Python 3 and pip manually."
            exit 1
            ;;
    esac

    if ! command -v python3 >/dev/null 2>&1; then
        error "Python installation failed. Please try installing it manually."
        exit 1
    fi

    success "Python has been installed."
    info "Please re-run this script to continue with the Audio Studio Pro installation."
    exit 0
}

install_package() {
    info "Installing/Updating Audio Studio Pro from GitHub..."
    python3 -m pip install --upgrade --force-reinstall "git+https://github.com/YaronKoresh/audio-studio-pro.git"
    
    if [ $? -ne 0 ]; then
        info "Standard installation failed, likely due to permissions. Retrying as Administrator..."
        ensure_sudo
        python3 -m pip install --upgrade --force-reinstall "git+https://github.com/YaronKoresh/audio-studio-pro.git"
        if [ $? -ne 0 ]; then
            error "Installation failed even with administrator privileges."
            exit 1
        fi
    fi
    success "Installation complete."
}

launch_app() {
    info "Launching Audio Studio Pro..."
    printf '\n'
    if [ -f "$HOME/.local/bin/audio-studio-pro" ]; then
        "$HOME/.local/bin/audio-studio-pro"
    else
        audio-studio-pro
    fi
    printf '\n'
    info "Audio Studio Pro has been closed."
}

main() {
    info "Checking for Python installation..."
    if ! command -v python3 >/dev/null 2>&1; then
        install_python
    else
        info "Python found."
        install_package
        launch_app
    fi
}

if [ "$1" = "--sudo-run" ]; then
    install_package
    info "Dependencies installed. Please run the script again without sudo to launch the application."
    exit 0
else
    main
fi